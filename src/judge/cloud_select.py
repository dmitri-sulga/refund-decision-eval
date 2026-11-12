"""Which models the cloud branch runs: docs/analysis_plan_models.md, Amendment 3, section 2.

The rules are code, applied to a snapshot of public listings taken before any request of the branch: OpenRouter's
model list (GET /api/v1/models), the provider endpoints of each model that passes the model-level rules (GET
/api/v1/models/{id}/endpoints) and, for each listed open-weights model, the parameter count that Hugging Face reports
for its weights (GET https://huggingface.co/api/models/{repo}?expand[]=safetensors). None of these requests needs a
key or sends anything about this study. The snapshot is committed under results/models/cloud/snapshot/, so anyone
can re-run the rules on it and get the registered list (`python -m judge.cloud_select --check`).

python -m judge.cloud_select --fetch     # take the snapshot (network, no key); done once, on 2026-11-12
python -m judge.cloud_select --resolve   # print what the rules select from the committed snapshot
python -m judge.cloud_select --check     # fail unless that equals the list registered in judge.cloud
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from . import config as C

SNAPSHOT = C.RESULTS / "models" / "cloud" / "snapshot"
MODELS_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{id}/endpoints"
HF_URL = "https://huggingface.co/api/models/{repo}?expand[]=safetensors&expand[]=cardData"

# ------------------------------------------------------------------ the rules (Amendment 3, section 2)
MIN_PARAMETERS = 32e9            # larger than every model run so far (gemma4:31b, 31.7B; qwen3.8:27b, 27.8B)
EXCLUDED_DEVELOPERS = ("anthropic", "openai", "openrouter")
# not a general chat model, a third-party derivative, or abliterated / uncensored (matched on the id, lower case)
EXCLUDED_ID_WORDS = ("code", "coder", "devstral", "guard", "safety", "-mt", "translat", "ocr", "schematron",
                     "ui-tars", "roleplay", "-rp", "abliterat", "uncensor", "venice", "thinking", "reasoning")
# a derivative of another model, as the listing describes it: "built on" or "based on" followed by a linked model
# name, or "fine-tune of" / "fine-tuned from" / "merge of" (matched on the description, lower case). "Built on a
# hybrid ... architecture" is not a derivative.
DERIVATIVE_PATTERN = re.compile(r"(built|based) on \[|fine-?tune[d]? (of|from)|merge of")
# weights already run (Amendment 1 and the study): a free variant of one of these is a serving-contrast reference row
ALREADY_RUN = {"google/gemma-4-31B-it": "gemma4:31b"}
MAX_FREE = 5                     # the free arm's daily request limit allows at most this many models in the schedule
MAX_PAID = 3
PAID_PROMPT_MAX = 1.00e-6        # USD per token on the chosen endpoint (1.00 USD per million)
PAID_COMPLETION_MAX = 4.00e-6    # USD per token (4.00 USD per million)
GOOD_PRECISION = ("fp32", "bf16", "fp16", "fp8")      # known, and fp8 or wider
NEEDED_PAID = ("structured_outputs", "temperature", "max_tokens")
# the basis of every estimate: tokens of 600 requests in the hosted arm (Amendment 2, as run: claude-haiku-5-5's
# 964,520 input tokens, the larger of the measured bases, and its 40,400 output tokens), with a 30% margin
BASIS_INPUT, BASIS_OUTPUT, MARGIN = 964_520, 40_400, 1.30


def endpoint_file(model_id: str) -> str:
    return re.sub(r"[/:]", "__", model_id) + ".json"


def _get(url: str) -> tuple[dict, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "refund-decision-eval"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8")), {"date": r.headers.get("date"), "status": r.status}


# ------------------------------------------------------------------ model-level rules
def text_only_out(m: dict) -> bool:
    a = m.get("architecture") or {}
    return "text" in (a.get("input_modalities") or []) and (a.get("output_modalities") or []) == ["text"]


def is_free(m: dict) -> bool:
    p = m.get("pricing") or {}
    return m["id"].endswith(":free") and float(p.get("prompt") or 0) == 0 and float(p.get("completion") or 0) == 0


def parameters(m: dict, hf: dict) -> int | None:
    rec = hf.get(m.get("hugging_face_id") or "") or {}
    return rec.get("total") if isinstance(rec.get("total"), int) else None


def derivative(m: dict, hf: dict) -> bool:
    desc = (m.get("description") or "").lower()
    if DERIVATIVE_PATTERN.search(desc):
        return True
    rec = hf.get(m.get("hugging_face_id") or "") or {}
    base = rec.get("base_model") or []
    base = [base] if isinstance(base, str) else base
    org = (m.get("hugging_face_id") or "").split("/")[0].lower()
    return any(isinstance(b, str) and b.split("/")[0].lower() != org for b in base)


def why_not(m: dict, hf: dict) -> str | None:
    """The first model-level rule a listed model fails (common to both arms), or None. The rule on parameters is
    applied by the arms (the serving-contrast row is below it)."""
    dev = m["id"].split("/")[0].lower()
    mid = m["id"].lower()
    if dev in EXCLUDED_DEVELOPERS:
        return "developer excluded (Anthropic, OpenAI or a router)"
    if not m.get("hugging_face_id"):
        return "no open weights listed (no Hugging Face id)"
    if parameters(m, hf) is None:
        return "Hugging Face reports no parameter count for its weights"
    if not text_only_out(m):
        return "does not take text and return only text"
    if (m.get("reasoning") or {}).get("mandatory"):
        return "reasoning cannot be switched off (listed as mandatory)"
    word = next((w for w in EXCLUDED_ID_WORDS if w in mid), None)
    if word:
        return f"not a general chat model, or abliterated / uncensored (id contains '{word}')"
    if derivative(m, hf):
        return "a derivative of another developer's model"
    return None


# ------------------------------------------------------------------ endpoint-level rules
def projected_cost(prompt: float, completion: float, request: float = 0.0) -> float:
    """USD for 600 requests at the estimate basis, with the margin."""
    return MARGIN * (BASIS_INPUT * prompt + BASIS_OUTPUT * completion + 600 * request)


def _price(e: dict, k: str) -> float:
    return float((e.get("pricing") or {}).get(k) or 0)


def paid_endpoint_ok(e: dict) -> bool:
    sp = e.get("supported_parameters") or []
    return (e.get("status") == 0 and all(p in sp for p in NEEDED_PAID)
            and (e.get("quantization") or "").lower() in GOOD_PRECISION
            and _price(e, "prompt") <= PAID_PROMPT_MAX and _price(e, "completion") <= PAID_COMPLETION_MAX
            and _price(e, "request") == 0)


def choose_endpoint(endpoints: list[dict], free: bool) -> dict | None:
    """Free arm: the model's endpoint with status 0 (the cheapest at the estimate if there are several; all are free).
    Paid arm: among endpoints that pass paid_endpoint_ok, the one with the lowest projected cost of 600 requests; a
    tie goes to the wider precision, then to the tag in alphabetical order."""
    ok = [e for e in endpoints if (e.get("status") == 0 if free else paid_endpoint_ok(e))]
    if not ok:
        return None
    rank = {q: i for i, q in enumerate(GOOD_PRECISION)}
    return min(ok, key=lambda e: (projected_cost(_price(e, "prompt"), _price(e, "completion"), _price(e, "request")),
                                  rank.get((e.get("quantization") or "").lower(), 99), e.get("tag") or ""))


def endpoint_summary(e: dict) -> dict:
    sp = e.get("supported_parameters") or []
    return {"provider_name": e.get("provider_name"), "tag": e.get("tag"), "quantization": e.get("quantization"),
            "prompt_usd_per_token": _price(e, "prompt"), "completion_usd_per_token": _price(e, "completion"),
            "request_usd": _price(e, "request"), "max_completion_tokens": e.get("max_completion_tokens"),
            "supports": {p: p in sp for p in ("structured_outputs", "response_format", "seed", "temperature",
                                              "reasoning", "max_tokens")}}


# ------------------------------------------------------------------ the two arms
def resolve(models: list[dict], hf: dict, endpoints: dict) -> dict:
    """The selection, from the snapshot. endpoints: {model id: [endpoint dicts]}."""
    rejected, free, reference, paid_pool = {}, [], [], []
    for m in models:
        reason = why_not(m, hf)
        n = parameters(m, hf)
        if is_free(m):
            if reason is None and m.get("hugging_face_id") in ALREADY_RUN:
                reference.append(m)
                continue
            if reason is None and n <= MIN_PARAMETERS:
                reason = f"{n / 1e9:.1f}B parameters, not larger than every model already run (32B)"
            if reason is None:
                free.append(m)
            else:
                rejected[m["id"]] = reason
        elif ":" not in m["id"] and not is_free(m) and reason is None and n > MIN_PARAMETERS:
            paid_pool.append(m)
    free.sort(key=lambda m: (-parameters(m, hf), m["id"]))
    for m in free[MAX_FREE:]:
        rejected[m["id"]] = f"beyond the first {MAX_FREE} free models by parameters"
    free = free[:MAX_FREE]
    in_free = {m["hugging_face_id"] for m in free + reference}
    out_free, out_ref = [], []
    for group, out in ((free, out_free), (reference, out_ref)):
        for m in group:
            e = choose_endpoint(endpoints.get(m["id"]) or [], free=True)
            if e is None:
                rejected[m["id"]] = "no endpoint with status 0"
                continue
            out.append(_row(m, hf, e, "free"))
    # paid: weights not in the free arm or already run; one model per developer, the largest (ties: the newest)
    best: dict[str, tuple] = {}
    for m in paid_pool:
        if m["hugging_face_id"] in in_free or m["hugging_face_id"] in ALREADY_RUN:
            continue
        e = choose_endpoint(endpoints.get(m["id"]) or [], free=False)
        if e is None:
            continue
        dev = m["id"].split("/")[0].lower()
        cand = (parameters(m, hf), m.get("created") or 0, m, e)
        if dev not in best or cand[:2] > best[dev][:2]:
            best[dev] = cand
    ranked = sorted(best.values(), key=lambda c: (-c[0], -c[1], c[2]["id"]))
    paid = [_row(m, hf, e, "paid") for _, _, m, e in ranked[:MAX_PAID]]
    # priced at 0 but without ':free' (Correction, m3): not free models by the rule, listed with the first rule they fail
    zero_price = {m["id"]: why_not(m, hf) or "no ':free' variant suffix" for m in models if not is_free(m)
                  and float((m.get("pricing") or {}).get("prompt") or 0) == 0
                  and float((m.get("pricing") or {}).get("completion") or 0) == 0}
    return {"free": out_free, "reference": out_ref, "paid": paid,
            "paid_next": [_row(m, hf, e, "paid")["id"] for _, _, m, e in ranked[MAX_PAID:MAX_PAID + 5]],
            "free_rejected": rejected, "zero_price_not_free": zero_price}


def _row(m: dict, hf: dict, e: dict, arm: str) -> dict:
    es = endpoint_summary(e)
    return {"id": m["id"], "arm": arm, "canonical_slug": m.get("canonical_slug"),
            "hugging_face_id": m.get("hugging_face_id"), "parameters": parameters(m, hf),
            "created": m.get("created"), "reasoning": m.get("reasoning"), "endpoint": es,
            "projected_usd": round(projected_cost(es["prompt_usd_per_token"], es["completion_usd_per_token"],
                                                  es["request_usd"]), 4)}


# ------------------------------------------------------------------ the snapshot
def load_snapshot(root=None) -> tuple[list[dict], dict, dict]:
    root = root or SNAPSHOT
    models = json.loads((root / "models.json").read_text())["data"]
    hf = json.loads((root / "huggingface.json").read_text())["models"]
    endpoints = {}
    for m in models:
        p = root / "endpoints" / endpoint_file(m["id"])
        if p.exists():
            endpoints[m["id"]] = (json.loads(p.read_text()).get("data") or {}).get("endpoints") or []
    return models, hf, endpoints


def fetch(root=None) -> dict:
    """Take the snapshot (network, no key, nothing about the study sent). Writes the raw responses."""
    root = root or SNAPSHOT
    (root / "endpoints").mkdir(parents=True, exist_ok=True)
    log = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "requests": []}
    raw, h = _get(MODELS_URL)
    (root / "models.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False) + "\n")
    log["requests"].append({"url": MODELS_URL, **h})
    models = raw["data"]
    hf = {}
    for m in models:
        repo = m.get("hugging_face_id")
        if not repo or repo in hf or not text_only_out(m):
            continue
        url = HF_URL.format(repo=urllib.parse.quote(repo, safe="/"))
        try:
            j, hh = _get(url)
            card = j.get("cardData") or {}
            hf[repo] = {"total": (j.get("safetensors") or {}).get("total"),
                        "parameters": (j.get("safetensors") or {}).get("parameters"),
                        "license": card.get("license"), "base_model": card.get("base_model")}
            log["requests"].append({"url": url, **hh})
        except Exception as exc:                  # noqa: BLE001 - recorded; the model then has no parameter count
            hf[repo] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            log["requests"].append({"url": url, "error": hf[repo]["error"]})
        time.sleep(0.25)
    (root / "huggingface.json").write_text(json.dumps({"models": hf}, indent=1, ensure_ascii=False) + "\n")
    for m in models:
        if why_not(m, hf) is None or (is_free(m) and m.get("hugging_face_id")):
            url = ENDPOINTS_URL.format(id=m["id"])
            try:
                j, hh = _get(url)
                (root / "endpoints" / endpoint_file(m["id"])).write_text(json.dumps(j, indent=1) + "\n")
                log["requests"].append({"url": url, **hh})
            except Exception as exc:              # noqa: BLE001
                log["requests"].append({"url": url, "error": f"{type(exc).__name__}: {exc}"[:200]})
            time.sleep(0.25)
    log["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (root / "fetch_log.json").write_text(json.dumps(log, indent=1) + "\n")
    return log


def snapshot_sha256(root=None) -> str:
    root = root or SNAPSHOT
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.json")):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--resolve", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        log = fetch()
        print(f"snapshot taken: {len(log['requests'])} requests, {log['started_at']} to {log['finished_at']}")
        return
    sel = resolve(*load_snapshot())
    if a.resolve:
        print(json.dumps(sel, indent=1))
        return
    problems = check(sel)
    if problems:
        sys.exit("cloud selection check failed:\n- " + "\n- ".join(problems))
    print("the registered list and endpoints are what the rules select from the committed snapshot")


def check(sel: dict) -> list[str]:
    """The registered models, in order, and for each its endpoint tag, provider, prices, seed support, format field
    and max_completion_tokens, against what the rules select from the snapshot (Correction, m4)."""
    from . import cloud
    got = [r["id"] for r in sel["free"] + sel["reference"] + sel["paid"]]
    if got != list(cloud.MODELS):
        return [f"the rules select {got}; registered {list(cloud.MODELS)}"]
    problems = []
    for r in sel["free"] + sel["reference"] + sel["paid"]:
        ep, reg = r["endpoint"], cloud.ENDPOINT[r["id"]]
        fmt = "json_schema" if ep["supports"]["structured_outputs"] else "json_object" if \
            ep["supports"]["response_format"] else None
        found = {"tag": ep["tag"], "provider": ep["provider_name"], "quantization": ep["quantization"] or "unknown",
                 "prompt": ep["prompt_usd_per_token"], "completion": ep["completion_usd_per_token"],
                 "seed": ep["supports"]["seed"], "format": fmt, "max_completion_tokens": ep["max_completion_tokens"]}
        for f, v in found.items():
            same = abs(v - reg[f]) <= 1e-15 if f in ("prompt", "completion") else v == reg[f]
            if not same:
                problems.append(f"{r['id']}: {f} is {v} in the snapshot, {reg[f]} registered")
    return problems


if __name__ == "__main__":
    main()
