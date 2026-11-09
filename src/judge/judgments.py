"""Turn cached model responses into predictions.

Pre-registered handling of invalid output (not valid JSON, or no boolean
decision): counted and reported; scored as 'no relief' in agreement metrics;
never automated (confidence NaN).
"""
from __future__ import annotations

import json
import math

from .llm import Cache, token_probability
from .prompts import DECISION_KEY


def parse(rec: dict) -> dict:
    out = {"valid": False, "pred": 0, "p_token": math.nan, "p_verbal": math.nan,
           "fields": {}, "wall_s": rec.get("wall_s"), "eval_count": rec.get("eval_count"),
           "prompt_eval_count": rec.get("prompt_eval_count")}
    try:
        obj = json.loads(rec["content"])
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(obj, dict) or not isinstance(obj.get(DECISION_KEY), bool):
        return out
    out["valid"] = True
    out["fields"] = obj
    out["pred"] = int(obj[DECISION_KEY])
    pv = obj.get("p_relief")
    if isinstance(pv, (int, float)) and not isinstance(pv, bool) and math.isfinite(pv):
        out["p_verbal"] = min(1.0, max(0.0, float(pv)))
    pt = token_probability(rec["content"], rec.get("logprobs"), DECISION_KEY)
    if pt is not None:
        out["p_token"] = pt
    return out


def load(version: str, item_ids, cache: Cache | None = None) -> dict:
    """{item_id: parsed} for every item; raises if any response is missing."""
    cache = cache or Cache()
    out = {}
    for i in item_ids:
        rec = cache.records.get(Cache.key(version, i))
        if rec is None:
            raise KeyError(f"no cached response for {version}|{i}")
        out[i] = parse(rec)
    return out


def available(version: str, item_ids, cache: Cache | None = None) -> bool:
    cache = cache or Cache()
    return all(Cache.key(version, i) in cache.records for i in item_ids)
