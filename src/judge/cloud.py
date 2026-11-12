"""Judge v1 and v2 on larger open-weights models through OpenRouter: docs/analysis_plan_models.md, Amendment 3.

Added after the held-out run, after the blind labels were locked and after both arms of Amendments 1 and 2 had run and
been reported. Sends only DEV and labelled-sample complaints and refuses any held-out ID (the local runner's item sets
and guard). Every request carries the frozen messages of judge v1 or v2 (prompts.py) and, where the pinned endpoint
supports it, the frozen JSON schema. Its cache, journal, run log and spend journal live under results/models/cloud/;
nothing is written anywhere else.

The API key is read from a file (--key-file, or OPENROUTER_KEY_FILE, default ~/.openrouter_key, which must not be
readable by group or others) when a request is sent, and only then. For the paid arm it must be a key dedicated to this
branch with a credit limit set in OpenRouter (Amendment 3, Correction, B1): its remaining limit, read from
GET /api/v1/key, is the server-side guard; the caps in this file are the second one. The key is sent in one header that
is never forwarded on a redirect (redirects are not followed), and it is never written to a file, a log line or an
error message: errors are rebuilt from the response body with the key (held in memory only after a request read it)
and anything shaped like a key removed.

Corrected three times before any request (Amendment 3, Correction, Correction 2 and Correction 3). Correction 2: a 403
never closes a model by itself (a spent key limit stops the arm; any other 403 without moderation `reasons`, a
provider's content filter, is a failure of that one request, recorded and sent again once on a later run); `make check`
accepts a model over its 5 USD cap only by its journaled overrun; the paid arm needs a key limited to exactly 15 USD
whose reading reports every field it relies on; a free run on another key than the registered one is a recorded
deviation. Correction 3 (C18-C21): errors are sorted by the structured fields the documentation says to branch on, never
by message text: a 403 is a content-filter refusal only when `metadata.error_type` is `content_policy_violation` or
`refusal`, and any other 403 (a permission, a guardrail block, a page that is not the API's JSON) stops the run as an
infrastructure or permission condition that closes no model; a budget stop of the arm only when `metadata.limit_source`
is `openrouter_key_limit` (a 403 or a 402); a second filter refusal of a request counts only at least 24 hours after the
first by the server's dates; a paid model whose remaining worst case the key's remaining limit cannot cover is closed by
the budget, so the queue behind it can go on.

python -m judge.cloud --verify                          # the checks `make check` runs; no network
python -m judge.cloud --estimate                        # projected and worst-case cost; no network
python -m judge.cloud --status                          # the key's limit and the free-model daily count (no model)
python -m judge.cloud --check-listings                  # the live listing check of every model (public; no key)
python -m judge.cloud --watch-listing MODEL             # the listing check once an hour (public; no key), see s. 9
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
QUARANTINE = DIR / "quarantine.jsonl"             # answers from another model or provider: kept, never scored
REGISTRATION = DIR / "registration.json"
LOCK = DIR / "run.lock"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
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
# USD per token, what the endpoint supports (which decides seed and the format field), and the longest output the
# listing allows (the bound of one answer billed beyond max_tokens; Correction, M1)
ENDPOINT = {
    "nvidia/nemotron-3-ultra-550b-a55b:free": {"tag": "nvidia", "provider": "Nvidia", "quantization": "unknown",
                                               "prompt": 0.0, "completion": 0.0, "seed": True, "format": None,
                                               "max_completion_tokens": 65536},
    "dots-studio/dots-3-note-preview:free": {"tag": "atlas-cloud/fp8", "provider": "AtlasCloud", "quantization": "fp8",
                                             "prompt": 0.0, "completion": 0.0, "seed": False, "format": "json_schema",
                                             "max_completion_tokens": 460800},
    "nvidia/nemotron-3.5-super-vl:free": {"tag": "nvidia", "provider": "Nvidia", "quantization": "unknown",
                                          "prompt": 0.0, "completion": 0.0, "seed": True, "format": "json_schema",
                                          "max_completion_tokens": 131072},
    "google/gemma-4-31b-it:free": {"tag": "google-ai-studio", "provider": "Google AI Studio", "quantization": "unknown",
                                   "prompt": 0.0, "completion": 0.0, "seed": True, "format": "json_object",
                                   "max_completion_tokens": 32768},
    "deepseek/deepseek-v4-pro-0813": {"tag": "novita/fp8", "provider": "Novita", "quantization": "fp8",
                                      "prompt": 0.99e-6, "completion": 2.97e-6, "seed": True, "format": "json_schema",
                                      "max_completion_tokens": 393216},
    "moonshotai/kimi-k2.6": {"tag": "crusoe/bf16", "provider": "Crusoe", "quantization": "bf16",
                             "prompt": 0.70e-6, "completion": 3.50e-6, "seed": True, "format": "json_schema",
                             "max_completion_tokens": 235929},
    "xiaomi/mimo-v2.6-pro": {"tag": "gmicloud/bf16", "provider": "GMICloud", "quantization": "bf16",
                             "prompt": 0.32625e-6, "completion": 0.6525e-6, "seed": True, "format": "json_schema",
                             "max_completion_tokens": 945000},
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
CREDIT_MARGIN_USD = 1.00                              # a paid model starts only with this much key limit to spare

# ------------------------------------------------------------------ pace and errors (Amendment 3, section 9)
MIN_INTERVAL_S = {"free": 4.0, "paid": 1.0}           # 15 per minute on free models, under their 20
FREE_PER_MINUTE = 20
FREE_DAILY_MARGIN = 25                                # free sends left unused of the day's remaining count
FREE_DAILY_FALLBACK = 950                             # if /api/v1/key does not report the count: journal-based cap
RETRY_DELAYS_S = (30, 60, 120, 240)
RETRY_AFTER_MAX_S = 300
MAX_ERRORS_PER_MODEL = 150                            # waited-out error answers only
REPEAT_GAP_S = 24 * 3600
WATCH_INTERVAL_S = 3600                               # --watch-listing: one public listing check an hour
MAX_LISTING_GAP_S = 2 * 3600                          # a failing streak counts only if no two checks are further apart
LOCK_RETRY_S = 30
# run_log events that record only a public listing check (no request, no key): they may be uncommitted when a model
# starts (Correction, m10)
LISTING_EVENTS = ("listing", "endpoint_status_changed", "listing_watch")
TRANSIENT_STATUSES = (408, 429, 500, 502, 503, 504, 520, 522, 524, 529)
ACCOUNT_STATUSES = (401, 402)
# a refusal caused by the account's own settings, not by the request: privacy / data policy, guardrails
ACCOUNT_PHRASES = ("data policy", "guardrail", "privacy setting")
# Correction 2, C11 (api-reference/errors and limits, read 2026-11-12): a spent key, guardrail or workspace budget is a
# 403 "Budget limit exceeded" with metadata.limit_source; a 402 with limit_source openrouter_in_flight_budget is
# temporary (Retry-After); a provider's content filter is a 403 with metadata.error_type and no `reasons`
IN_FLIGHT_SOURCE = "openrouter_in_flight_budget"
# Correction 3, C18-C19 (api-reference/errors and limits, read again 2026-11-12 15:26 UTC): the documentation says to
# "branch on `limit_source`, not on this text"; a 402 or 403 with limit_source openrouter_key_limit is the branch key's
# own limit spent (the registered cap: exactly 15 USD, never reset), the only budget stop of an arm. A provider's
# content filter is a 403 whose metadata.error_type is one of FILTER_TYPES; `permission_denied`, a guardrail block or a
# body that is not the API's JSON is not. `token_limit_exceeded` (400) is "a token budget enforced by OpenRouter (e.g.
# credit-based cap)", an account condition, not a rejection.
KEY_LIMIT_SOURCE = "openrouter_key_limit"
FILTER_TYPES = ("content_policy_violation", "refusal")
ACCOUNT_TYPES = ("token_limit_exceeded",)
FILTER_TRIES = 2                                      # a request refused by a content filter: final after two refusals
#                                                       at least REPEAT_GAP_S apart by the server's dates (C20)
NOT_SERVED = ("transient", "rejected", "account", "quota", "refused", "mismatch", "filtered", "key_limit", "forbidden")
# a key, or a key's label as GET /api/v1/key shows it (the key's first characters, three dots, its last ones:
# Correction 2, C17)
KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9_\-]*\.{3}[A-Za-z0-9_\-]*|sk-or-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{20,}")
DONE_REASON = {"stop": "stop", "length": "length"}    # every other finish_reason is kept as it is (malformed)
VERDICT_WORDS = {"rejected": "the request was rejected",
                 "account": "an account condition (the key, the credit balance or the account's privacy settings)",
                 "quota": "the free models' daily request limit",
                 "transient": "a rate limit, an overload or a server error, after its retries",
                 "unavailable": "the model or its pinned endpoint is no longer listed as registered",
                 "mismatch": "an answer from another model or provider (kept in quarantine, not scored)",
                 "key_limit": "the branch key's limit in OpenRouter was spent (a budget stop of the arm)",
                 "filtered": "the provider's content filter refused the request (a deviation of that request)",
                 "forbidden": ("a 403 that is neither the branch key's spent limit nor a content filter (a permission, "
                               "a guardrail block or a page that is not the API's JSON): an infrastructure or "
                               "permission condition")}
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


_KEYS_READ: set[str] = set()      # the keys this process read to send a request (memory only), for scrub()


def read_key() -> str:
    """Called only when a request is sent."""
    why = key_file_ok()
    if why:
        raise CloudError(None, why + "; nothing was sent.")
    with open(key_path(), encoding="utf-8") as f:
        k = f.read().strip()
    if not k:
        raise CloudError(None, "the key file is empty; nothing was sent.")
    _KEYS_READ.add(k)
    return k


def scrub(text) -> str:
    """Removes the key this process sent (if any request was sent) and anything shaped like a key. Never reads the
    key file (Correction, m1)."""
    s = str(text)
    for k in _KEYS_READ:
        s = s.replace(k, "[redacted]")
    return KEY_PATTERN.sub("[redacted]", s)


class CloudError(Exception):
    """A failed request, rebuilt without the request (no headers, no key)."""

    def __init__(self, status, message, request_id=None, etype=None, maybe_billed=False, retry_after=None,
                 usage=None, date=None, remaining=None, moderation=None, in_200=False, limit=None, limit_source=None,
                 raw=None):
        # `own_message` is OpenRouter's own message; `message` adds the provider's raw payload for the record. Rules
        # that read text read only `own_message` (Correction 3, C19: the provider's raw text never sorts an error)
        self.own_message = scrub(message)[:500]
        full = str(message) + (f" ({str(raw)[:200]})" if raw else "")
        self.status, self.message, self.request_id, self.etype = status, scrub(full)[:500], request_id, etype
        self.maybe_billed, self.retry_after, self.usage, self.date = maybe_billed, retry_after, usage, date
        self.remaining, self.moderation, self.in_200, self.limit = remaining, moderation, in_200, limit
        self.limit_source = limit_source
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
    assert body["max_tokens"] == MAX_TOKENS          # the hard per-request output limit, on every request
    return ep["prompt"] * input_upper(body) + ep["completion"] * body["max_tokens"]


def overrun_headroom(tag: str) -> float:
    """Correction, M1: the most one answer can cost beyond its reservation if the provider bills output past
    max_tokens, up to the endpoint's listed max_completion_tokens. Kept free under the branch cap before every paid
    request, so one such answer cannot take the branch over 15 USD; the model is closed after it."""
    ep = ENDPOINT[tag]
    return ep["completion"] * max(0, ep["max_completion_tokens"] - MAX_TOKENS)


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
    meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
    moderation = meta if "reasons" in meta else None
    code = err.get("code") if isinstance(err.get("code"), int) else status
    raw = meta.get("raw")
    message = str(err.get("message") or "")
    try:
        retry_after = float(hdrs.get("retry-after")) if hdrs and hdrs.get("retry-after") else None
    except ValueError:
        retry_after = None
    # Correction 2, C11: the documented error type is metadata.error_type; error.type is kept only as a fallback
    etype = meta.get("error_type") or err.get("type")
    limit_source = meta.get("limit_source")
    return CloudError(code, message, hdrs.get("x-generation-id") if hdrs else None,
                      str(etype) if etype is not None else None,
                      retry_after=retry_after, usage=usage, date=hdrs.get("date") if hdrs else None,
                      remaining=hdrs.get("x-ratelimit-remaining") if hdrs else None, moderation=moderation,
                      in_200=in_200, limit=hdrs.get("x-ratelimit-limit") if hdrs else None,
                      limit_source=str(limit_source) if limit_source is not None else None, raw=raw)


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
    usage = None
    try:
        doc = json.loads(raw)
        err = doc["error"]
        assert isinstance(err, dict)
        # Correction 3, C21 (the review's m-6): an HTTP error that reports usage is charged by it, as C12 says
        usage = _valid_usage(doc.get("usage"))
    except Exception:                            # noqa: BLE001 - not the API's JSON (a proxy's or a CDN's page)
        err = {"message": raw[:300] or str(exc.reason)}
    e = _error_from_json(err, exc.code, exc.headers, usage=usage)
    e.status = exc.code                          # the HTTP status decides; error.code may repeat it
    return e


def per_minute(exc: CloudError) -> bool:
    """A 429 with X-RateLimit-Remaining 0 can be the per-minute limit as well as the daily one (the headers describe
    the limit that was hit; api-reference/limits, read 2026-11-12). It is the per-minute one when X-RateLimit-Limit is
    the per-minute allowance (20) or the message names a minute; otherwise it is taken as the daily limit, which only
    pauses the run (Correction, m2)."""
    try:
        if exc.limit is not None and int(str(exc.limit).strip()) == FREE_PER_MINUTE:
            return True
    except ValueError:
        pass
    return bool(re.search(r"per[- ]?min|per minute|/min", exc.own_message, re.I))   # OpenRouter's own text (C18)


def classify(exc: BaseException, tag: str) -> tuple[str, str]:
    """Amendment 3, section 9, as corrected (Correction 2, C11, and Correction 3, C18-C21). Only structured fields
    sort an error into a stop of the arm or a deviation; OpenRouter's own message (never the provider's raw payload)
    is read only for the registered account phrases and the per-minute 429.
    'moderated': OpenRouter's moderation flag, an error whose metadata gives `reasons` (an answer, malformed, scored no
    money back). 'key_limit': a 402 or 403 whose metadata.limit_source is openrouter_key_limit, the branch key's own
    limit spent (stops the arm; never closes a model by itself). 'transient': 408, 429, 5xx and the 402 whose
    limit_source is openrouter_in_flight_budget, waited out. 'account': 401, any other 402, a 402 or 403 with any other
    limit_source (a guardrail or workspace budget, the account's balance: the owner's to change), a 400
    `token_limit_exceeded`, a 4xx whose own message names the data policy, a privacy setting or a guardrail, or nothing
    sent. 'quota': the free models' daily limit (stops; resumes after the reset). 'filtered': a 403 whose
    metadata.error_type is content_policy_violation or refusal (a provider's content filter: a failure of that one
    request, recorded as a deviation; never a rejection). 'forbidden': any other 403 (`permission_denied`, a guardrail
    block, a page that is not the API's JSON): stops the run like an account condition, closes no model, never a
    rejection. 'rejected': any other 3xx or 4xx. 'lost': anything else."""
    if not isinstance(exc, CloudError):
        return "lost", scrub(f"{type(exc).__name__}: {exc}")[:500]
    s = exc.status
    if exc.moderation is not None:
        return "moderated", exc.message
    if s in (402, 403) and exc.limit_source == KEY_LIMIT_SOURCE:
        return "key_limit", exc.message
    if s == 402 and exc.limit_source == IN_FLIGHT_SOURCE:
        return "transient", exc.message
    if s in (402, 403) and exc.limit_source is not None:
        return "account", exc.message
    if s == 429 and tag in FREE and str(exc.remaining) == "0" and not per_minute(exc):
        return "quota", exc.message
    if s == 403 and exc.etype in FILTER_TYPES:
        return "filtered", exc.message
    if s in ACCOUNT_STATUSES or (s is not None and 400 <= s < 500
                                 and any(p in exc.own_message.lower() for p in ACCOUNT_PHRASES)) \
            or (s == 400 and exc.etype in ACCOUNT_TYPES):
        return "account", exc.message
    if s == 403:
        return "forbidden", exc.message
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


# reasoning written into the answer's text (Correction, M4): a <think>, <thinking> or <reasoning> block, closed or not
THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>(.*?)(</\1>|$)", re.S | re.I)


def reasoning_in_content(content) -> int:
    """Characters of reasoning text inside the answer's own text."""
    if not isinstance(content, str):
        return 0
    return sum(max(1, len(m.group(2))) for m in THINK_BLOCK.finditer(content))


def reasoning_seen(resp: dict) -> tuple[int, int, int]:
    """(reasoning tokens reported, characters of reasoning text returned in message.reasoning(_details), characters
    of reasoning text inside message.content)."""
    usage = resp.get("usage") or {}
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    msg = resp["choices"][0]["message"]
    text = msg.get("reasoning") or ""
    details = msg.get("reasoning_details") or []
    chars = len(text) if isinstance(text, str) else 0
    chars += sum(len(str(d.get("text") or d.get("summary") or "")) for d in details if isinstance(d, dict))
    return int(rt) if isinstance(rt, int) else 0, chars, reasoning_in_content(msg.get("content"))


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def provider_matches(tag: str, name) -> bool:
    """Correction, M2: the reported provider is compared after case folding and removing spaces and punctuation, with
    the snapshot's provider name and with the provider part of the pinned tag (`atlas-cloud/fp8` -> atlascloud). A
    missing provider field is accepted by an explicit rule: the request pins the endpoint with fallbacks off, so
    OpenRouter cannot route it elsewhere; such answers are counted and reported per cell."""
    if name is None:
        return True
    ep = ENDPOINT[tag]
    return _norm(name) in {_norm(ep["provider"]), _norm(ep["tag"].split("/")[0])}


def answer_record(tag, version, item_id, body, rhash, resp, headers, requested_at, wall, drop) -> dict:
    usage = resp["usage"]
    choice = resp["choices"][0]
    finish = choice.get("finish_reason")
    rt, rchars, cchars = reasoning_seen(resp)
    return {
        "key": key(tag, version, item_id), "model": tag, "prompt_version": version, "item_id": item_id,
        "request_sha256": rhash, "response_model": resp["model"], "provider": resp.get("provider"),
        "generation_id": headers.get("generation-id"), "server_date": headers.get("date"),
        "requested_at": requested_at, "wall_s": round(wall, 3), "fields_dropped": sorted(drop),
        "endpoint_tag": ENDPOINT[tag]["tag"], "content": choice["message"].get("content") or "",
        "finish_reason": finish, "native_finish_reason": choice.get("native_finish_reason"),
        "done_reason": DONE_REASON.get(finish, finish), "reasoning_tokens": rt, "reasoning_chars": rchars + cchars,
        "reasoning_in_content_chars": cchars, "usage": usage, "prompt_eval_count": usage["prompt_tokens"], "eval_count": usage["completion_tokens"],
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
    """The waited-out error answers that count toward MAX_ERRORS_PER_MODEL. The in-flight 402 is not one of them
    (Correction 3, C21, the review's m-5): it reports the account's settling spend, not the model or its endpoint."""
    return sum(1 for e in journal if e.get("kind") == "transient" and split_key(e["key"])[0] == tag
               and e.get("limit_source") != IN_FLIGHT_SOURCE)


def free_sends_today(journal) -> int:
    day = now()[:10]
    return sum(1 for e in journal if e.get("kind") in ("send", "resend", "retry") and e["at"][:10] == day
               and split_key(e["key"])[0] in FREE)


def last_answer_s(tag: str, cache=None) -> float | None:
    """When the model's last cached answer was requested (local clock), or None: a success that ends any failing
    streak before it (Correction, B2)."""
    cache = cache if cache is not None else {r["key"]: r for r in _jsonl(CACHE_FILE)}
    times = [H._seconds(r["requested_at"]) for k, r in cache.items() if split_key(k)[0] == tag and r.get("requested_at")]
    return max(times) if times else None


def unavailable_streak(tag: str, run_log: list[dict], cache=None) -> list[dict]:
    """Correction, B2: the failing listing checks of the model since its last passing check and since its last answer
    (no intervening success), in order."""
    after = last_answer_s(tag, cache)
    streak: list[dict] = []
    for e in run_log:
        if e.get("model") != tag or e.get("event") != "listing":
            continue
        if e.get("ok"):
            streak = []
        elif after is None or H._seconds(e["at"]) > after:
            streak.append(e)
    return streak


def streak_closes(streak: list[dict]) -> bool:
    """A failing streak closes a model only if it spans at least 24 hours by the server's dates and no two consecutive
    checks in it are more than MAX_LISTING_GAP_S apart (so the moment of a check cannot pick the result: Correction,
    B2; `--watch-listing` checks once an hour). Only the latest unbroken run of dated checks counts: a check without
    a server date breaks the run like a gap, and does not block every later run (Correction 2, C17)."""
    times = [H._server_seconds(e.get("server_date")) for e in streak]
    if len(times) < 2 or times[-1] is None:
        return False
    start = len(times) - 1                      # the latest run of dated checks with no gap over MAX_LISTING_GAP_S
    while start > 0 and times[start - 1] is not None and 0 <= times[start] - times[start - 1] <= MAX_LISTING_GAP_S:
        start -= 1
    return times[-1] - times[start] >= REPEAT_GAP_S


def repeated(tag: str, run_log: list[dict], journal: list[dict], cache=None) -> dict | None:
    """Section 9, condition (c), as corrected (Correction, B2): the model's last run stopped on a rejection or on
    'unavailable', and with no answer of the model in between, either the same rejection (same request, status and
    error type) was observed at least 24 hours apart by the server's date headers, or the listing check failed on
    every check of a streak that spans 24 hours with checks at most two hours apart. Returns that last stop, or None."""
    runs = [e for e in run_log if e.get("model") == tag and e.get("event") in ("run_started", "run_stopped")]
    last = runs[-1] if runs and runs[-1]["event"] == "run_stopped" else None
    if not last or last.get("verdict") not in ("rejected", "unavailable"):
        return None
    if last["verdict"] == "unavailable":
        return last if streak_closes(unavailable_streak(tag, run_log, cache)) else None
    after = last_answer_s(tag, cache)
    same = [e for e in journal if e.get("key") == last.get("key") and e.get("kind") == "rejected"
            and e.get("status") == last.get("status") and e.get("error_type") == last.get("error_type")
            and (after is None or H._seconds(e["at"]) > after)]
    times = [t for t in (H._server_seconds(e.get("server_date")) for e in same) if t is not None]
    return last if len(times) >= 2 and max(times) - min(times) >= REPEAT_GAP_S else None


def load_registration() -> dict:
    if not REGISTRATION.exists():
        raise Refusal(f"results/models/cloud/{REGISTRATION.name} is missing; refusing.")
    reg = json.loads(REGISTRATION.read_text())
    need = ("repository", "code_commit", "code_published_at", "code_published_source", "gist_id", "gist_url",
            "gist_created_at", "gist_file_sha256", "key_reading")
    missing = [k for k in need if not reg.get(k)]
    if missing:
        raise Refusal(f"results/models/cloud/{REGISTRATION.name} lacks {', '.join(missing)}: the branch's code "
                      "commit, the time GitHub records for its publication and its source, the secret gist with its "
                      "files' hashes, and the key's reading (`--status`) must be recorded before the first request.")
    return reg


def github_main_head() -> str:
    import subprocess
    slug = load_registration()["repository"]
    out = subprocess.run(["gh", "api", f"repos/{slug}/git/refs/heads/main", "--jq", ".object.sha"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


# ------------------------------------------------------------------ live listings (no key)
def listing_check(tag: str, getter=public_get) -> tuple[str | None, str | None, int | None]:
    """(None or why the model is not available as registered, the server's date, the pinned endpoint's status). The
    model must still be listed under its id with its Hugging Face id, the pinned endpoint must be listed with the same
    provider and the same or wider precision, and a price no higher than the registered one (zero for a free model).
    The endpoint's status is recorded, not required (Correction, B2: it changed from 0 to -2 for crusoe/bf16 within an
    hour of the snapshot); a request to an endpoint that is down fails and is handled by the rules of section 9."""
    from . import cloud_select as S
    snap = {m["id"]: m for m in S.load_snapshot()[0]}
    models, date = getter(MODELS_URL)
    live = {m["id"]: m for m in models.get("data") or []}
    if tag not in live:
        return f"{tag} is no longer listed", date, None
    if live[tag].get("hugging_face_id") != snap[tag].get("hugging_face_id"):
        return f"{tag} is now listed with other weights ({live[tag].get('hugging_face_id')})", date, None
    eps, date = getter(ENDPOINTS_URL.format(id=tag))
    ep = ENDPOINT[tag]
    match = [e for e in (eps.get("data") or {}).get("endpoints") or [] if e.get("tag") == ep["tag"]]
    if not match:
        return f"the pinned endpoint {ep['tag']} of {tag} is no longer listed", date, None
    e = match[0]
    status = e.get("status")
    if _norm(e.get("provider_name")) != _norm(ep["provider"]):
        return f"the pinned endpoint {ep['tag']} is now served by {e.get('provider_name')}", date, status
    if (e.get("quantization") or "unknown") != ep["quantization"]:
        return f"the pinned endpoint {ep['tag']} now reports quantization {e.get('quantization')}", date, status
    p = e.get("pricing") or {}
    if float(p.get("prompt") or 0) > ep["prompt"] + 1e-15 or float(p.get("completion") or 0) > ep["completion"] + 1e-15 \
            or float(p.get("request") or 0) > 0:
        return f"the pinned endpoint {ep['tag']} of {tag} now costs more than registered", date, status
    # Correction 2, C17: the room kept for one overrun answer (C3) rests on the registered max_completion_tokens
    mct = e.get("max_completion_tokens")
    if tag not in FREE and (not _num(mct) or mct > ep["max_completion_tokens"]):   # _num: finite (C21)
        return (f"the pinned endpoint {ep['tag']} of {tag} now lists max_completion_tokens {mct}, not at most the "
                f"registered {ep['max_completion_tokens']} on which the overrun room rests"), date, status
    return None, date, status


def record_listing(tag: str, why, date, status, context: str) -> None:
    """Every listing check goes to the run log with the server's date and the endpoint's status; a status different
    from the last one recorded for the model (or from the snapshot's 0) is also recorded as a flip (Correction, B2)."""
    prev = [e.get("endpoint_status") for e in _jsonl(RUN_LOG) if e.get("model") == tag and e.get("event") == "listing"
            and e.get("endpoint_status") is not None]
    before = prev[-1] if prev else 0
    log_event({"event": "listing", "model": tag, "ok": why is None, "reason": why, "server_date": date,
               "endpoint_status": status, "context": context})
    if status is not None and status != before:
        log_event({"event": "endpoint_status_changed", "model": tag, "endpoint": ENDPOINT[tag]["tag"], "from": before,
                   "to": status, "server_date": date})


def check_listings(getter=public_get) -> list[dict]:
    """The listing check of every model, recorded (public requests, no key). Run on the start day before the first
    request (pre_run_checklist.md, G)."""
    out = []
    for tag in MODELS:
        why, date, status = listing_check(tag, getter)
        record_listing(tag, why, date, status, "check-listings")
        out.append({"model": tag, "ok": why is None, "reason": why, "endpoint_status": status, "server_date": date})
    return out


def watch_listing(tag: str, getter=public_get, sleep=time.sleep, max_checks: int | None = None) -> str:
    """Correction, B2: the listing check of one model once an hour (public, no key) until it passes or its failing
    streak can close the model (24 hours, no gap over two hours). Takes the run lock for each check only."""
    if tag not in MODELS:
        raise Refusal(f"{tag} is not a model of Amendment 3.")
    checks = 0
    while True:
        while True:
            try:
                acquire_lock()
                break
            except Refusal:
                sleep(LOCK_RETRY_S)
        try:
            why, date, status = listing_check(tag, getter)
            record_listing(tag, why, date, status, "watch")
            if why:
                log_event({"event": "run_stopped", "model": tag, "verdict": "unavailable", "reason": why})
        finally:
            release_lock()
        checks += 1
        if why is None:
            return f"{tag}: the listing check passes ({date}); resume with --run {tag}."
        if streak_closes(unavailable_streak(tag, _jsonl(RUN_LOG))):
            return f"{tag}: the listing check failed on every hourly check for 24 hours; --close-incomplete {tag} may close it."
        if max_checks is not None and checks >= max_checks:
            return f"{tag}: {checks} check(s), still failing ({why})."
        sleep(WATCH_INTERVAL_S)


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
    if dirty and not started and not only_pre_request_appends(dirty):
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
    if arm_stopped_by_key(tag, _jsonl(RUN_LOG)):
        raise Refusal(f"the branch key's limit in OpenRouter was spent during the {arm(tag)} arm; no request of that "
                      "arm is sent after that (Correction 2, C13).")
    from .label import is_locked
    if not is_locked("owner"):
        raise Refusal("the blind labels are not locked; refusing.")
    why = key_file_ok()
    if why:
        raise Refusal(why + "; nothing was sent.")
    why, date, status = listing_check(tag, getter)
    record_listing(tag, why, date, status, "run")
    if why:
        log_event({"event": "run_stopped", "model": tag, "verdict": "unavailable", "reason": why})
        raise Refusal(f"stopped before {tag}: {why}. Nothing is substituted; run again later. The model can be closed "
                      f"only after `--watch-listing {tag}` finds it failing on every hourly check for 24 hours with no "
                      "passing check or answer in between.")
    sets, test = X.item_sets()
    return {"sets": sets, "test": test, "registration": reg}


def arm_stopped_by_key(tag: str, run_log: list[dict]) -> bool:
    """Correction 2, C13: a 403 reporting the branch key's spent limit stopped the arm of `tag`."""
    return any(e.get("event") == "key_limit_stop" and e.get("arm") == arm(tag) for e in run_log)


# run_log and spend records written before any request of a model (listing checks, key readings, the estimate, a run
# that paused or was refused before its first send): they may be uncommitted when a model starts (Correction, m10, and
# Correction 2, C17)
PRE_REQUEST_EVENTS = LISTING_EVENTS + ("account_before", "account_after", "run_started", "daily_pause", "deviation")


def _appended(path, rel: str) -> list[dict] | None:
    """The records appended to `path` since HEAD, or None if the file is not HEAD's text plus appended lines."""
    try:
        committed = X.git("show", f"HEAD:{rel}")
    except Exception:                            # noqa: BLE001 - not committed yet
        committed = ""
    now_text = path.read_text() if path.exists() else ""
    if not now_text.startswith(committed):
        return None
    return [json.loads(ln) for ln in now_text[len(committed):].splitlines() if ln.strip()]


def only_pre_request_appends(dirty: list[str]) -> bool:
    """The only uncommitted changes are appends to the run log and the spend journal written before any request
    (listing checks, the key's readings, the estimate, a refusal or a pause before the first send), so a run that sent
    nothing does not force a commit and a publication before the model can be tried again."""
    files = {}
    for p in (RUN_LOG, SPEND):
        if p.is_relative_to(C.ROOT):
            files[p.relative_to(C.ROOT).as_posix()] = p
    if not dirty or any(d not in files for d in dirty):
        return False
    for d in dirty:
        added = _appended(files[d], d)
        if added is None:
            return False
        if files[d] == SPEND:
            ok = all(e.get("event") == "estimate" for e in added)
        else:
            ok = all((e.get("event") in PRE_REQUEST_EVENTS and not (e.get("event") == "daily_pause" and e.get("sends")))
                     or (e.get("event") == "run_stopped" and e.get("verdict") == "unavailable") for e in added)
        if not ok:
            return False
    return True


# the fields of GET /api/v1/key that are recorded (api-reference/limits, read 2026-11-12: limit "or null if
# unlimited", limit_remaining, limit_reset "null means it never resets", usage all time, usage_daily in the UTC day,
# is_free_tier, free_model_daily_requests {used, limit, remaining} in the current UTC day)
KEY_FIELDS = ("limit", "limit_remaining", "limit_reset", "include_byok_in_limit", "usage",
              "usage_daily", "is_free_tier", "free_model_daily_requests", "budgets", "effective_budget")
PAID_KEY_FIELDS = ("limit", "limit_remaining", "limit_reset")      # must be present in the reading (Correction 2, C15)


def account_readings(transport) -> dict:
    """The key's limit, its use and the free-model daily count, from GET /api/v1/key (key; no model is called, nothing
    is billed). GET /api/v1/credits is not used: it needs a management key (Correction, B1). The key's label, which
    shows part of the key, is not kept. A field the response leaves out is recorded as missing, not read as null
    (Correction 2, C15); the guardrail and workspace budgets are recorded too."""
    out = {}
    try:
        k = (transport.get(KEY_URL) or {}).get("data")
        k = k if isinstance(k, dict) else {}
        for f in KEY_FIELDS:
            out[f] = k.get(f)
        out["fields_missing"] = [f for f in KEY_FIELDS if f not in k]
        if not isinstance(out["free_model_daily_requests"], dict):
            out["free_model_daily_requests"] = None
    except Exception as exc:                     # noqa: BLE001
        out["key_error"] = scrub(f"{type(exc).__name__}: {exc}")[:200]
    return out


def _num(x) -> bool:
    """A finite number, not a bool: NaN and infinity (which json.loads accepts) are not readings (Correction 3, C21,
    the review's m-2)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def other_budgets(before: dict) -> list[str]:
    """Correction 3, C19: the scopes of the budgets GET /api/v1/key lists for the key other than the key's own limit
    (a guardrail or workspace budget). The paid arm runs only under the key's own limit."""
    out = []
    for b in (before.get("budgets") if isinstance(before.get("budgets"), list) else []) + (
            [before["effective_budget"]] if isinstance(before.get("effective_budget"), dict) else []):
        scope = b.get("scope") if isinstance(b, dict) else None
        if scope != "key_limit" and str(scope) not in out:
            out.append(str(scope))
    return out


def key_short(before: dict, worst: float) -> bool:
    """Correction 3, C19 (the review's MAJOR-3, R4): the key is configured as registered but its remaining limit, a
    finite number, does not cover the model's remaining worst case plus CREDIT_MARGIN_USD. The key's limit cannot be
    raised (exactly 15 USD, never reset: C15), so this is a budget stop that closes the model, as the caps of section 7
    close one before it starts; the next model of the queue may then start."""
    rem = before.get("limit_remaining")
    return paid_key_config_refusal(before) is None and _num(rem) and rem < worst + CREDIT_MARGIN_USD


def paid_key_refusal(tag: str, before: dict, worst: float) -> str | None:
    """Correction, B1, as corrected by Correction 2, C15, and Correction 3, C19: a paid model starts only on a key
    whose limit is set in OpenRouter to exactly the branch cap (15 USD), that never resets (`limit_reset` present and
    null), under no guardrail or workspace budget, with at least the model's remaining worst case plus 1 USD left.
    None if it may. A refusal for the remaining limit alone (`key_short`) closes the model; any other is resumable."""
    why = paid_key_config_refusal(before)
    if why:
        return why
    rem = before.get("limit_remaining")
    if not _num(rem) or rem < worst + CREDIT_MARGIN_USD:
        return (f"the key reports {rem} USD of its limit left, less than the worst case of the remaining requests "
                f"({worst:.4f} USD) plus {CREDIT_MARGIN_USD:.2f}")
    return None


def paid_key_config_refusal(before: dict) -> str | None:
    """The key's configuration as C1, C15 and C19 require, without its remaining limit. None if it holds."""
    if before.get("key_error"):
        return f"GET /api/v1/key failed ({before['key_error']})"
    missing = [f for f in PAID_KEY_FIELDS if f in (before.get("fields_missing") or ())]
    if missing:
        return (f"GET /api/v1/key did not report {', '.join(missing)}; a missing field is not read as null "
                "(\"unlimited\" or \"never resets\")")
    if before.get("limit") is None:
        return ("the key has no credit limit (GET /api/v1/key reports limit null): the paid arm needs a key dedicated "
                f"to this branch with a limit of exactly {CAP_USD:.0f} USD set in OpenRouter")
    if not _num(before["limit"]) or abs(before["limit"] - CAP_USD) > 1e-9:
        return f"the key's limit is {before['limit']} USD, not exactly the branch cap of {CAP_USD:.0f} USD"
    if before.get("limit_reset") is not None:
        return f"the key's limit resets ({before['limit_reset']}); it must never reset"
    others = other_budgets(before)
    if others:
        return (f"GET /api/v1/key lists a budget other than the key's own limit ({', '.join(others)}); the paid arm "
                "runs only under the key's own limit of exactly 15 USD (remove the guardrail or workspace budget)")
    return None


def key_differs(before: dict, reading) -> list[str]:
    """Correction 2, C15: the fields of the registered `key_reading` (the branch key, read once with `--status`) that
    the reading before this run does not match; a free run on another key is recorded as a deviation."""
    if before.get("key_error"):
        return ["key_error"]
    if not isinstance(reading, dict):
        return ["key_reading"]
    out = [f for f in ("limit", "limit_reset", "is_free_tier") if f in reading and before.get(f) != reading[f]]
    want = (reading.get("free_model_daily_requests") or {}).get("limit")
    if want is not None and (before.get("free_model_daily_requests") or {}).get("limit") != want:
        out.append("free_model_daily_requests.limit")
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


def note_overrun(k, es, r, c, reserve, body, quarantined=False) -> None:
    """Correction, M1 (C3), and Correction 2, C14: journals an overrun (`kind: overrun` in sent.jsonl) with the tokens,
    the cost and the reservation; `r` is None for a charge without an answer."""
    tag = split_key(k)[0]
    out_over = bool(r and r["eval_count"] > MAX_TOKENS)
    in_over = bool(r and r["prompt_eval_count"] > input_upper(body))
    ov = {"at": now(), "key": k, "kind": "overrun", "completion_tokens": r["eval_count"] if r else None,
          "max_tokens": MAX_TOKENS, "prompt_tokens": r["prompt_eval_count"] if r else None,
          "input_bound": input_upper(body), "cost_usd": round(c, 10), "reservation_usd": round(reserve, 10),
          **({"quarantined": True} if quarantined else {}), **({"without_answer": True} if r is None else {})}
    _append(JOURNAL, ov)
    es.append(ov)
    log_event({"event": "overrun", "model": tag, "key": k, "output_over": out_over, "input_over": in_over,
               "cost_over_reservation": c > reserve + 1e-12, "quarantined": quarantined, "without_answer": r is None})


def filtered_by_key(journal: list[dict], tag: str) -> dict[str, list[dict]]:
    """Correction 2, C12: the content-filter refusals of each request of the model, in order."""
    out: dict[str, list[dict]] = {}
    for e in journal:
        if e.get("kind") == "filtered" and split_key(e["key"])[0] == tag:
            out.setdefault(e["key"], []).append(e)
    return out


def filtered_counts(journal: list[dict], tag: str) -> dict[str, int]:
    return {k: len(v) for k, v in filtered_by_key(journal, tag).items()}


def filter_final(refusals: list[dict]) -> bool:
    """Correction 3, C20 (the review's MAJOR-1, R2): a request's content-filter refusal is final only when a second
    refusal comes at least REPEAT_GAP_S (24 hours) after the first by the server's date headers, as the rejection rule
    of section 9 requires (Correction, B2), so that the moment of a run cannot pick the result. Refusals without a
    server date do not count toward it."""
    times = [t for t in (H._server_seconds(e.get("server_date")) for e in refusals) if t is not None]
    return len(times) >= FILTER_TRIES and max(times) - min(times) >= REPEAT_GAP_S


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
        if committed + est["projected_usd"] > CAP_USD or mine + est["projected_usd"] > SUB_CAP_USD:
            scope = "branch" if committed + est["projected_usd"] > CAP_USD else "model"
            log_event({"event": "budget_stop", "model": tag, "scope": scope, "when": "before the model",
                       "spent_usd": round(committed, 6), **est})
            close_model(tag, "not run: projection exceeded remaining budget" if not any(
                split_key(k)[0] == tag for k in st["cache"]) else "stopped by the cap: the projection did not fit")
            raise Refusal(f"{tag}: the projection {est['projected_usd']:.4f} USD does not fit under the "
                          f"{'branch' if scope == 'branch' else 'model'} cap; nothing was sent, and it is closed.")
        why = paid_key_refusal(tag, before, est["worst_case_usd"])
        if why and key_short(before, est["worst_case_usd"]):
            # Correction 3, C19 (the review's MAJOR-3, R4): the key's remaining limit cannot cover this model, and the
            # limit cannot be raised: a budget stop that closes the model, so the queue behind it is not stuck
            log_event({"event": "budget_stop", "model": tag, "scope": "key", "when": "before the run",
                       "limit_remaining_usd": before.get("limit_remaining"),
                       "worst_case_usd": est["worst_case_usd"], "margin_usd": CREDIT_MARGIN_USD})
            reason = close_model(tag, f"stopped by the budget: the key's remaining limit in OpenRouter "
                                      f"({before.get('limit_remaining')} USD) did not cover the worst case of the "
                                      f"model's remaining requests ({est['worst_case_usd']:.4f} USD) plus "
                                      f"{CREDIT_MARGIN_USD:.2f}")
            raise Refusal(f"{tag}: {why}; nothing was sent, and it is closed ({reason}). The next model of the paid "
                          "queue may be tried.")
        if why:
            raise Refusal(f"{tag}: {why}; nothing was sent.")
    else:
        # Correction 2, C15: a free run checks that it runs on the registered key; if not, it is a recorded deviation
        differs = key_differs(before, (pre.get("registration") or {}).get("key_reading"))
        if differs:
            log_event({"event": "deviation", "kind": "key_differs", "model": tag, "fields": differs})
            print(f"{tag}: the key's reading differs from the registered key_reading in {', '.join(differs)}; "
                  "recorded as a deviation (the report lists it).", flush=True)
    allowance = _free_allowance(tag, before, st["journal"])
    log_event({"event": "run_started", "model": tag, "resend_unanswered": resend_unanswered,
               "free_sends_allowed": allowance})
    drop = dropped_fields(tag)
    done, sends, last = 0, 0, [0.0]
    # Correction 2, C12, and Correction 3, C20: content-filter refusals of earlier runs; a request whose refusal is
    # final (two refusals at least 24 hours apart by the server's dates) is not sent again
    filtered_before = filtered_by_key(st["journal"], tag)

    def record_spend(entry):
        _append(SPEND, {"at": now(), "model": tag, **entry,
                        "running_total_usd": round(spend_state()["computed"] + (entry.get("cost_usd") or 0), 6)})

    def stop(k, verdict, message, he):
        log_event({"event": "run_stopped", "model": tag, "key": k, "verdict": verdict,
                   "status": he.status if he else None, "error_type": he.etype if he else None,
                   "reason": message, "request_id": he.request_id if he else None})

    for version, split, item_id, rec, narrative in work_list(tag, sets, records):
        k = key(tag, version, item_id)
        if k in st["cache"] or filter_final(filtered_before.get(k, [])):
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
            headroom = overrun_headroom(tag)
            if committed + reserve + headroom > CAP_USD or mine + reserve > SUB_CAP_USD:
                scope = "branch" if committed + reserve + headroom > CAP_USD else "model"
                log_event({"event": "budget_stop", "model": tag, "key": k, "scope": scope,
                           "spent_usd": round(committed, 6), "model_spent_usd": round(mine, 6),
                           "next_worst_case_usd": round(reserve, 6), "overrun_headroom_usd": round(headroom, 6)})
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
                # an error inside a 200 on a paid model is charged at its worst case when it reports no usage: the
                # provider may have billed it (Correction 2, C17); any error that reports usage is charged by it, as
                # C12 says (Correction 3, C21, the review's m-6)
                amount = None
                if verdict == "lost" or (he and (he.usage or (he.in_200 and tag not in FREE))):
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
                           "in_200": bool(he and he.in_200),
                           **({"limit_source": he.limit_source} if he and he.limit_source else {})}
                    _append(JOURNAL, err)
                    es.append(err)
                if amount is not None and amount > reserve + 1e-12:
                    # Correction 2, C14: a charge above the reservation is an overrun as C3 defines it
                    note_overrun(k, es, None, amount, reserve, body)
                    if tag not in FREE:
                        stop(k, verdict, message, he)
                        close_model(tag, f"{k}: billed a cost above the reservation for an answer that could not be "
                                         "used; the cap no longer rests on the reservation")
                        raise Refusal(f"{k}: charged {amount:.6f} USD against a reservation of {reserve:.6f}; "
                                      f"{tag} is closed as incomplete.") from None
                if verdict == "key_limit":
                    # Correction 2, C13: a budget stop of the arm; it closes no model by itself
                    log_event({"event": "key_limit_stop", "model": tag, "arm": arm(tag), "key": k,
                               "status": he.status, "limit_source": he.limit_source, "reason": message[:300]})
                    stop(k, verdict, message, he)
                    raise Refusal(f"{tag}: the branch key's limit in OpenRouter is spent (HTTP {he.status}: "
                                  f"{message[:200]}). No request of the {arm(tag)} arm is sent after this; no model "
                                  "is closed by it. `--close-incomplete` records a model of this arm that is not "
                                  "closed as stopped by the budget.") from None
                if verdict == "filtered":
                    # Correction 2, C12: a failure of this one request, recorded as a deviation; the run goes on, and
                    # the request is sent once more on a later run
                    n_f = len(filtered_before.get(k, [])) + 1
                    log_event({"event": "filtered", "model": tag, "key": k, "attempt": n_f,
                               "status": he.status, "error_type": he.etype, "reason": message[:300]})
                    print(f"[{tag} {version} {split}] {item_id} refused by the provider's content filter "
                          f"(HTTP {he.status} {he.etype}), refusal {n_f}; it is final only at a second refusal "
                          "at least 24 hours after the first. The run goes on.",
                          flush=True)
                    break
                if verdict == "rejected" and he is not None:
                    field = first_request_drop(he, tag, [x for v in journal.values() for x in v])
                    if field and field not in drop:
                        log_event({"event": "field_dropped", "model": tag, "field": field, "message": message[:300]})
                        drop.append(field)
                        kind, retries = "send", 0
                        continue
                if verdict == "transient":
                    if he is None or he.limit_source != IN_FLIGHT_SOURCE:
                        errors += 1                  # the in-flight 402 is not an error answer (C21)
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
            if not model_matches(tag, r["response_model"]) or not provider_matches(tag, r["provider"]):
                # Correction, M2: charged, kept in quarantine (never scored), the run stops; the request is sent
                # again on the next run. A second such answer of the model closes it.
                record_spend({"event": "charged", "key": k, "request_id": r["generation_id"], "cost_usd": round(c, 10),
                              "note": f"answer served by {r['response_model']} at {r['provider']}",
                              "reservation": rid_})
                _append(QUARANTINE, r)
                mm = {"at": now(), "key": k, "kind": "mismatch", "model_answered": r["response_model"],
                      "provider": r["provider"]}
                _append(JOURNAL, mm)
                es.append(mm)
                if c > reserve + 1e-12:
                    # Correction 2, C14: a quarantined answer billed above its reservation is an overrun too (C3)
                    note_overrun(k, es, r, c, reserve, body, quarantined=True)
                    if tag not in FREE:
                        close_model(tag, f"{k}: billed a cost above the reservation (an answer kept in quarantine); "
                                         "the cap no longer rests on the reservation")
                        raise Refusal(f"{k}: the quarantined answer cost {c:.6f} USD against a reservation of "
                                      f"{reserve:.6f}; {tag} is closed as incomplete.")
                n_mm = sum(1 for e in _jsonl(JOURNAL) if e.get("kind") == "mismatch" and split_key(e["key"])[0] == tag)
                if n_mm >= 2:
                    close_model(tag, f"served by {r['response_model']} at {r['provider']} (a second answer from "
                                     "another model or provider)")
                    raise Refusal(f"the API answered {k} with {r['response_model']} at {r['provider']}, the second "
                                  f"such answer of {tag}; {tag} is closed as incomplete.")
                stop(k, "mismatch", f"served by {r['response_model']} at {r['provider']}", None)
                raise Refusal(f"the API answered {k} with {r['response_model']} at {r['provider']}, not the "
                              f"registered model at {ENDPOINT[tag]['provider']}; the answer is kept in quarantine and "
                              f"charged, not scored. Resume with --run {tag}; a second such answer closes the model.")
            over_out = r["eval_count"] > MAX_TOKENS
            over_in = r["prompt_eval_count"] > input_upper(body)
            over_cost = c > reserve + 1e-12
            record_spend({"event": "answer", "key": k, "request_id": r["generation_id"],
                          "prompt_tokens": r["usage"]["prompt_tokens"],
                          "completion_tokens": r["usage"]["completion_tokens"],
                          "reasoning_tokens": r["reasoning_tokens"], "reported_cost_usd": r["usage"].get("cost"),
                          "cost_usd": round(c, 10), "reservation": rid_,
                          **({"over_reservation": True} if over_cost else {})})
            _append(CACHE_FILE, r)
            st["cache"][k] = r
            done += 1
            print(f"[{tag} {version} {split}] {item_id} {r['wall_s']:.1f}s finish={r['finish_reason']} "
                  f"in={r['prompt_eval_count']} out={r['eval_count']} reasoning={r['reasoning_tokens']} "
                  f"total={spend_state()['computed']:.4f} USD", flush=True)
            if over_out or over_in or over_cost:
                # Correction, M1: a deviation type of its own, journaled; the answer is kept and scored as it is.
                # On a paid model the cap no longer rests on the reservation, so the model is closed.
                note_overrun(k, es, r, c, reserve, body)
                if tag not in FREE:
                    what = ("output above max_tokens" if over_out else "input above the bound" if over_in
                            else "a cost above the reservation")
                    close_model(tag, f"{k}: billed {what}; the cap no longer rests on the reservation")
                    raise Refusal(f"{k}: billed {what} ({r['prompt_eval_count']} in, {r['eval_count']} out, "
                                  f"{c:.6f} USD against a reservation of {reserve:.6f}); {tag} is closed as "
                                  "incomplete.")
            break
    expected = expected_keys(tag, sets)
    missing = expected - set(st["cache"])
    if not missing:
        log_event({"event": "model_finished", "model": tag, "answers": len(expected)})
        write_model_total(tag)
        print(f"{tag}: all {len(expected)} answers are in ({spend_state()['computed']:.4f} USD so far). Commit and "
              "publish results/models/cloud/ before the next model.")
    else:
        filt = filtered_by_key(_jsonl(JOURNAL), tag)
        final = {k for k in missing if filter_final(filt.get(k, []))}
        again = sorted(k for k in missing if filt.get(k) and k not in final)
        if again:
            # Correction 2, C12, and Correction 3, C20: each request refused by a content filter is sent again by a
            # later run, once per run, until it is answered or refused again at least 24 hours after its first refusal
            print(f"{tag}: {len(again)} request(s) refused by the provider's content filter and not yet refused again "
                  "24 hours later: " + ", ".join(again[:10]) + f". Each is sent once more by a later --run {tag}; a "
                  "refusal is final only at least 24 hours after the first by the server's dates; nothing was closed.",
                  flush=True)
            return {"model": tag, "new_answers": done, "missing": len(missing), "filtered_to_send_again": len(again),
                    "spent_usd": round(spend_state()["computed"], 6)}
        if not all(resend_served(journal.get(k, [])) for k in missing - final):
            raise Refusal(f"{tag}: a missing request has not had its one resend; nothing was closed.")
        parts = ([f"{len(missing - final)} request(s) unanswered after their one resend"] if missing - final else []) + (
            [f"{len(final)} request(s) refused by the provider's content filter on 2 runs at least 24 hours apart"]
            if final else [])
        close_model(tag, "; ".join(parts))
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
        branch_stop = tag not in FREE and any(e.get("event") == "budget_stop" and e.get("scope") == "branch"
                                              for e in run_log)
        key_stop = arm_stopped_by_key(tag, run_log)
        if not (branch_stop or key_stop) and not any(e.get("model") == tag and e.get("event") in (
                "run_started", "run_stopped") for e in run_log):
            raise Refusal(f"{tag} has not started; there is nothing to close.")
        if any(e.get("event") == "budget_stop" and e.get("model") == tag for e in run_log):
            reason = "stopped by the cap"
        elif key_stop:
            # Correction 2, C13: the arm stopped at the branch key's limit; a budget stop, never a rejection
            reason = "stopped by the budget: the branch key's limit in OpenRouter was spent"
        elif branch_stop:
            reason = "stopped by the cap: the branch reached its 15 USD cap"
        elif error_count(st["journal"], tag) >= MAX_ERRORS_PER_MODEL:
            reason = "error-answer limit reached"
        else:
            last = repeated(tag, run_log, st["journal"])
            if last is None:
                raise Refusal(f"{tag} cannot be closed: none of the conditions of section 9 holds (a budget stop, the "
                              "error-answer limit, the same rejection at least 24 hours apart by the server's dates "
                              "with no answer in between, or the listing check failing on every check of 24 hours "
                              f"with no gap over two hours and no success in between). Resume with --run {tag}.")
            how = ("on every listing check for at least 24 hours, no two checks more than two hours apart"
                   if last["verdict"] == "unavailable" else "observed at least 24 hours apart")
            reason = f"{VERDICT_WORDS[last['verdict']]}, {how} ({last.get('key') or tag}: {last.get('reason')})"
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
        # Correction 2, C12, and Correction 3, C20: a request is not sent again once its content-filter refusal is
        # final (a second refusal at least 24 hours after the first by the server's dates)
        refusals, final_at = [], None
        for n, e in enumerate(entries):
            if e.get("kind") == "filtered":
                refusals.append(e)
                if filter_final(refusals):
                    final_at = n
                    break
        if final_at is not None and any(e.get("kind") in ("send", "resend", "retry") for e in entries[final_at + 1:]):
            problems.append(f"journal: {k} was sent again after its content-filter refusal became final (two "
                            "refusals at least 24 hours apart)")
        # Correction 3, C18-C19: only the structured fields sort an error into a filter refusal or a key-limit stop
        for e in entries:
            if e.get("kind") == "filtered" and (e.get("status") != 403 or e.get("error_type") not in FILTER_TYPES):
                problems.append(f"journal: {k} records a content-filter refusal whose status and error type "
                                f"({e.get('status')}, {e.get('error_type')}) are not a content filter's")
            if e.get("kind") == "key_limit" and e.get("limit_source") != KEY_LIMIT_SOURCE:
                problems.append(f"journal: {k} records a key-limit stop without limit_source {KEY_LIMIT_SOURCE}")
    for tag in MODELS:
        if error_count(journal, tag) > MAX_ERRORS_PER_MODEL:
            problems.append(f"journal: {tag} has more error answers than the rules allow")
    for e in run_log:
        if e.get("event") == "key_limit_stop" and e.get("limit_source") != KEY_LIMIT_SOURCE:
            problems.append(f"run_log: a key_limit_stop of the {e.get('arm')} arm without limit_source "
                            f"{KEY_LIMIT_SOURCE} (C19: a budget stop is read from the structured field only)")
    answers = {e["key"]: e for e in spend if e.get("event") == "answer"}
    overruns = {e["key"] for e in journal if e.get("kind") == "overrun"}
    reserved = {e["id"]: e["cost_usd"] for e in spend if e.get("event") == "reserved"}
    for e in spend:
        if e.get("event") in ("answer", "charged") and e.get("reservation") in reserved \
                and e["cost_usd"] > reserved[e["reservation"]] + 1e-9 and e["key"] not in overruns:
            problems.append(f"spend: {e['key']} cost more than its reservation and no overrun is journaled")
    for tag in PAID_QUEUE:
        mine = [e for e in journal if split_key(e["key"])[0] == tag]
        first = next((n for n, e in enumerate(mine) if e.get("kind") == "overrun"), None)
        if first is not None and any(e.get("kind") in ("send", "resend", "retry") for e in mine[first + 1:]):
            problems.append(f"journal: {tag} was sent a request after an overrun, which closes a paid model")
        if sum(e.get("kind") == "overrun" for e in mine) > 1:           # C21, the review's m-1
            problems.append(f"journal: {tag} has more than one overrun entry; its first overrun closes a paid model")
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
                                           or not provider_matches(tag, r.get("provider"))):
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
        if (r.get("eval_count") or 0) > MAX_TOKENS and k not in overruns:
            problems.append(f"cache: {k} reports more output tokens than max_tokens and no overrun is journaled")
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
        mine = committed_for(tag, spend)
        if mine <= SUB_CAP_USD + 1e-9:
            continue
        excess = overrun_excess(tag, journal, spend, reserved)
        if excess <= 0:
            problems.append(f"spend: {tag} is over its 5 USD cap and no journaled overrun accounts for it")
        elif mine - excess > SUB_CAP_USD + 1e-9:
            problems.append(f"spend: {tag} is over its 5 USD cap by more than its journaled overrun accounts for "
                            f"({mine:.6f} USD spent, {excess:.6f} USD of it beyond the overrun's reservation)")
    return problems


def overrun_excess(tag: str, journal: list[dict], spend: list[dict], reserved: dict) -> float:
    """Correction 2, C14: what the model's journaled overruns cost beyond their reservations, counting an overrun only
    if the spend journal holds the matching charge (same request, same cost) against a reservation of the amount the
    overrun records. A model over its 5 USD cap passes `make check` only if its spend less this is within the cap."""
    total, used = 0.0, set()
    for ov in journal:
        if ov.get("kind") != "overrun" or split_key(ov["key"])[0] != tag:
            continue
        # Correction 3, C21 (the review's m-1): each spend entry accounts for one overrun at most, so an overrun line
        # written twice cannot count its excess twice
        match = next((n for n, e in enumerate(spend) if n not in used and e.get("key") == ov["key"]
                      and e.get("event") in ("answer", "charged")
                      and abs(e["cost_usd"] - (ov.get("cost_usd") or 0)) <= 1e-9 and e.get("reservation") in reserved
                      and abs(reserved[e["reservation"]] - (ov.get("reservation_usd") or 0)) <= 1e-9), None)
        if match is not None:
            used.add(match)
            total += max(0.0, ov["cost_usd"] - ov["reservation_usd"])
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--run", metavar="MODEL")
    g.add_argument("--close-incomplete", metavar="MODEL")
    g.add_argument("--break-stale-lock", action="store_true")
    g.add_argument("--check-listings", action="store_true")
    g.add_argument("--watch-listing", metavar="MODEL")
    ap.add_argument("--resend-unanswered", action="store_true")
    ap.add_argument("--key-file", metavar="PATH", help=f"the key file (default: ${KEY_FILE_ENV}, else {DEFAULT_KEY_FILE})")
    a = ap.parse_args()
    if a.key_file:
        os.environ[KEY_FILE_ENV] = a.key_file
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
        elif a.check_listings:
            acquire_lock()
            try:
                print(json.dumps(check_listings(), indent=1))
            finally:
                release_lock()
        elif a.watch_listing:
            print(watch_listing(a.watch_listing))
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
