"""Ollama client with an append-only JSONL cache.

Every model response is cached under the key (prompt version, item id) together
with the SHA-256 of the exact request (model, messages, JSON schema, options).
A cached key whose request hash differs from the current one is an error, so a
prompt cannot change silently under an existing version name. With a complete
cache every report regenerates offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.request
from datetime import datetime, timezone

from . import config as C


# ------------------------------------------------------------------ transport
def _http(path: str, body: dict | None = None, timeout: float = 60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(C.OLLAMA_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def environment() -> dict:
    """Ollama version and the exact model digest (recorded with every response)."""
    version = _http("/api/version")["version"]
    digest = next(m["digest"] for m in _http("/api/tags")["models"] if m["name"] == C.MODEL)
    details = _http("/api/show", {"model": C.MODEL}).get("details", {})
    return {"ollama_version": version, "model": C.MODEL, "model_digest": digest,
            "model_details": details}


def request_body(messages: list, schema: dict) -> dict:
    return {"model": C.MODEL, "messages": messages, "format": schema, "think": False,
            "stream": False, "options": C.OPTIONS, "logprobs": True,
            "top_logprobs": C.TOP_LOGPROBS, "keep_alive": "60m"}


def request_hash(body: dict) -> str:
    canon = {k: body[k] for k in ("model", "messages", "format", "think", "options",
                                  "logprobs", "top_logprobs")}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ logprobs
BOOL_TOKENS = {"true", "false"}


def compact_logprobs(lp: list | None) -> dict | None:
    """Keep every chosen token with its logprob, and the top-k alternatives only
    at positions where a JSON boolean was a candidate (that is where decisions
    are made). Byte arrays are dropped. Keeps the cache small."""
    if not lp:
        return None
    tokens, top = [], {}
    for i, e in enumerate(lp):
        tokens.append([e["token"], round(e["logprob"], 5)])
        alts = e.get("top_logprobs") or []
        if any(a["token"].strip() in BOOL_TOKENS for a in alts):
            top[str(i)] = [[a["token"], round(a["logprob"], 5)] for a in alts
                           if a["logprob"] > -9000]
    return {"tokens": tokens, "top": top}


def token_probability(content: str, logprobs: dict | None, key: str) -> float | None:
    """P(key == true) from the token distribution at the position where the
    boolean value of `key` was generated, renormalised over {true, false}.
    Returns None when the position cannot be located."""
    if not logprobs:
        return None
    toks = [t for t, _ in logprobs["tokens"]]
    if "".join(toks) != content:
        return None
    m = re.search(r'"%s"\s*:\s*' % re.escape(key), content)
    if not m:
        return None
    # index of the token that contains the first character of the value
    pos, start = len(toks) - 1, 0
    for i, t in enumerate(toks):
        if start <= m.end() < start + len(t):
            pos = i
            break
        start += len(t)
    for j in (pos, pos - 1, pos + 1):
        alts = logprobs["top"].get(str(j))
        if not alts:
            continue
        lt = [lp for tok, lp in alts if tok.strip() == "true"]
        lf = [lp for tok, lp in alts if tok.strip() == "false"]
        if not lt and not lf:
            continue
        floor = min(lp for _, lp in alts) - 1.0  # unseen option: below the k-th best
        a = _logsumexp(lt) if lt else floor
        b = _logsumexp(lf) if lf else floor
        return 1.0 / (1.0 + math.exp(b - a))
    return None


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ------------------------------------------------------------------ cache
class Cache:
    def __init__(self, path=C.CACHE_FILE):
        self.path = path
        self.records: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r["key"] in self.records:
                            raise RuntimeError(f"duplicate cache key {r['key']}")
                        self.records[r["key"]] = r

    @staticmethod
    def key(version: str, item_id: str) -> str:
        return f"{version}|{item_id}"

    def get(self, key: str, rhash: str):
        r = self.records.get(key)
        if r is not None and r["request_sha256"] != rhash:
            raise RuntimeError(f"stale cache entry {key}: the request changed but the "
                               "prompt version did not; bump the version name")
        return r

    def put(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records[record["key"]] = record


def fetch(env: dict, version: str, item_id: str, body: dict, rhash: str) -> dict:
    """One model call; returns the cache record without storing it (thread-safe)."""
    t0 = time.time()
    resp = _http("/api/chat", body, timeout=C.REQUEST_TIMEOUT_S)
    wall = time.time() - t0
    return {
        "key": Cache.key(version, item_id), "prompt_version": version, "item_id": item_id,
        "request_sha256": rhash, "model": C.MODEL,
        "model_digest": env["model_digest"], "ollama_version": env["ollama_version"],
        "options": C.OPTIONS, "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_s": round(wall, 3), "content": resp["message"]["content"],
        "thinking": resp["message"].get("thinking"), "done_reason": resp.get("done_reason"),
        **{k: resp.get(k) for k in ("prompt_eval_count", "prompt_eval_cached_count", "eval_count",
                                    "total_duration", "load_duration", "prompt_eval_duration",
                                    "eval_duration")},
        "logprobs": compact_logprobs(resp.get("logprobs")),
    }


def call(cache: Cache, env: dict | None, version: str, item_id: str,
         messages: list, schema: dict) -> tuple[dict, bool]:
    """Return (record, was_cached). Calls the model only on a cache miss."""
    body = request_body(messages, schema)
    rhash = request_hash(body)
    key = Cache.key(version, item_id)
    hit = cache.get(key, rhash)
    if hit is not None:
        return hit, True
    if env is None:
        raise RuntimeError(f"cache miss for {key} in offline mode")
    rec = fetch(env, version, item_id, body, rhash)
    cache.put(rec)
    return rec, False
