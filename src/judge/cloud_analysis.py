"""Figures and labels of the cloud branch (docs/analysis_plan_models.md, Amendment 3), in its own table.

Written before any request of the branch. Applies the local extension's computations unchanged (models_analysis: the
malformed rule, the paired intervals, the three readings, the person's figures). The comparison set is the six models
registered in judge.cloud.COMPARED: 12 comparisons with the issuer lookup and 6 payout comparisons, fixed before any
request. A model of the set that reports any reasoning (a reasoning token, or reasoning text) leaves the set and is a
descriptive reference row; so is google/gemma-4-31b-it:free, the same weights as Amendment 1's gemma4:31b, whose
agreement with that local run is given (a serving contrast). Reads the branch's files, the local extension's cache and
the study's cache. Calls no model.

python -m judge.cloud_analysis            # writes results/models/cloud/analysis.json and prints table and text
"""
from __future__ import annotations

import json
import statistics

import numpy as np

from . import cloud as K
from . import config as C
from . import hosted as H
from . import hosted_analysis as HA
from . import metrics as M
from . import models as XM
from . import models_analysis as A
from .models import CELLS, REFERENCE, _jsonl

N_A = 2 * len(K.COMPARED)      # 12 comparisons with the issuer lookup (models x prompts), fixed before any request
N_B = len(K.COMPARED)          # 6 payout comparisons, fixed before any request
SERVING_ROW = "the same weights as gemma4:31b of Amendment 1, served by OpenRouter; not part of the comparisons"
MOVED_ROW = "reported reasoning (see Amendment 3, section 3); not part of the comparisons"
PERSON_NOTE = HA.PERSON_NOTE       # the same framing sentence as the hosted arm, from one source


def reasoning_off(model: str, answers: list[dict]) -> bool:
    """Section 3: in the comparison set only if registered there and no answer reports a reasoning token or returns
    reasoning text (one token is enough: the rule is mechanical and fixed in advance)."""
    return model in K.COMPARED and not any((r.get("reasoning_tokens") or 0) > 0 or (r.get("reasoning_chars") or 0) > 0
                                           for r in answers)


def costs(spend: list[dict]) -> dict:
    out = {}
    open_ = H.spend_state(spend)["open"]
    for t in K.MODELS:
        mine = [e for e in spend if e.get("model") == t and e.get("event") in ("answer", "charged")]
        ans = list({e["key"]: e for e in mine if e["event"] == "answer"}.values())
        reserved = [e for e in open_ if e.get("model") == t]
        out[t] = {"answers": len(ans), "charged_without_answer": len(mine) - len(ans),
                  "open_reservations": len(reserved),
                  **{f: sum(e.get(f) or 0 for e in ans) for f in ("prompt_tokens", "completion_tokens",
                                                                  "reasoning_tokens")},
                  "reported_usd": round(sum(e.get("reported_cost_usd") or 0 for e in ans), 6),
                  "usd": round(sum(e["cost_usd"] for e in mine) + sum(e["cost_usd"] for e in reserved), 6)}
    out["total_usd"] = round(sum(out[t]["usd"] for t in K.MODELS), 6)
    out["cap_usd"], out["sub_cap_usd"] = K.CAP_USD, K.SUB_CAP_USD
    return out


def account(run_log: list[dict]) -> dict:
    """The first and the last credit reading the runner recorded (the API's own figures)."""
    reads = [e for e in run_log if e.get("event") in ("account_before", "account_after")
             and isinstance(e.get("remaining_credits"), (int, float))]
    if not reads:
        return {}
    return {"first": {k: reads[0].get(k) for k in ("at", "total_credits", "total_usage", "remaining_credits")},
            "last": {k: reads[-1].get(k) for k in ("at", "total_credits", "total_usage", "remaining_credits")}}


def model_status(model, closed, answered, n, run_log) -> tuple[str, bool]:
    e = closed.get(model)
    if e and e["event"] == "model_finished":
        return "complete", True
    if e:
        return e["reason"], True
    stops = [x for x in run_log if x.get("model") == model and x.get("event") in ("run_started", "run_stopped",
                                                                                  "daily_pause")]
    last = ""
    if stops and stops[-1]["event"] == "run_stopped":
        last = f"; its last run stopped: {K.VERDICT_WORDS.get(stops[-1].get('verdict'), 'an interruption')}"
    elif stops and stops[-1]["event"] == "daily_pause":
        last = "; paused at the free models' daily limit"
    return (f"not closed ({answered} of {n} answered{last})" if answered or stops else "not started"), False


def analyse(cache_path=None, study_cache_path=None, local_cache_path=None, records=None, run_log=None,
            spend_path=None) -> dict:
    from .baselines import issuer_rate
    from .data import load_records, load_splits
    from .label import labels_path, load_labels

    records = records if records is not None else load_records()
    splits = load_splits()
    sets, _ = XM.item_sets()
    cache = {r["key"]: r for r in _jsonl(cache_path or K.CACHE_FILE)}
    study_cache = {r["key"]: r for r in _jsonl(study_cache_path or C.CACHE_FILE)}
    local_cache = {r["key"]: r for r in _jsonl(local_cache_path or XM.CACHE_FILE)}
    events = _jsonl(run_log or K.RUN_LOG)
    closed = {e["model"]: e for e in events if e.get("event") in ("closed_incomplete", "model_finished")}
    f, base, _ = issuer_rate(splits["train"], records)
    dev, lab = sets["dev"], sets["labelled"]
    y = np.array([records[i]["label"] for i in dev])
    lookup = np.array([int(f(records[i]) > base) for i in dev])
    person = load_labels(labels_path("owner"))
    out = {"arm": "cloud (Amendment 3)", "order": list(K.MODELS), "reference": REFERENCE[0],
           "record_pays_dev": int(y.sum()), "n_dev": len(dev), "lookup_pays_dev": int(lookup.sum()),
           "comparisons": {"A": N_A, "B": N_B}, "models": {}}
    ref = {}
    for version, split in CELLS:
        ids = dev if split == "dev" else lab
        ans = A.cell_answers(REFERENCE[0], version, ids, cache, study_cache)
        ref[(version, split)] = np.array([ans[i][0] for i in ids])
    n_all = len(dev) * 2 + len(lab) * 2
    for model in K.MODELS:
        mine = [r for k, r in cache.items() if k.startswith(model + "|")]
        status, legit = model_status(model, closed, len(mine), n_all, events)
        compared = reasoning_off(model, mine)
        m = {"status": status, "closed": legit, "compared": compared, "arm": K.arm(model),
             "endpoint": K.ENDPOINT[model]["tag"],
             "row_note": None if compared else SERVING_ROW if model in K.REFERENCE_ONLY else MOVED_ROW, "cells": {}}
        for version, split in CELLS:
            ids = dev if split == "dev" else lab
            ans = A.cell_answers(model, version, ids, cache, study_cache)
            c = {"n": len(ids), "answered": len(ans), "malformed": sum(v[1] for v in ans.values())}
            fins = [v[2].get("finish_reason") for v in ans.values()]
            c["finish_reasons"] = {s: fins.count(s) for s in sorted(set(fins), key=str)}
            c["moderated"] = sum(1 for v in ans.values() if v[2].get("moderated"))
            c["providers"] = sorted({str(v[2].get("provider")) for v in ans.values()})
            walls = [v[2]["wall_s"] for v in ans.values() if v[2].get("wall_s") is not None]
            if walls:
                c["seconds_per_call"] = {"median": statistics.median(walls), "mean": statistics.fmean(walls)}
            rt = [v[2].get("reasoning_tokens") or 0 for v in ans.values()]
            if rt:
                c["reasoning_tokens"] = {"median": statistics.median(rt), "max": max(rt), "total": sum(rt)}
            if len(ans) < len(ids):
                c["status"] = f"incomplete ({len(ans)} of {len(ids)} answered)"
                c["pays_answered"] = sum(v[0] for v in ans.values())
                m["cells"][f"{version}|{split}"] = c
                continue
            preds = np.array([ans[i][0] for i in ids])
            bad = np.array([ans[i][1] for i in ids])
            c["pays"] = int(preds.sum())
            c["kappa_with_reference"] = M.cohen_kappa(ref[(version, split)], preds)
            same = K.SAME_WEIGHTS_AS.get(model)
            if same:
                local = A.cell_answers(same, version, ids, local_cache, study_cache)
                if len(local) == len(ids):
                    lp = np.array([local[i][0] for i in ids])
                    c["serving_contrast"] = {"local_model": same, "kappa_with_local": M.cohen_kappa(lp, preds),
                                             "same_decision": int((lp == preds).sum()), "n": len(ids)}
            c["status"] = "not interpretable (format)" if c["malformed"] > A.FORMAT_LIMIT[split] else "complete"
            yy = y if split == "dev" else np.array([records[i]["label"] for i in ids])
            c["kappa_vs_record"] = M.bootstrap(yy, preds)["kappa"]
            if split == "dev" and c["status"] == "complete":
                if compared:
                    c["A"] = A.dev_cell(y, lookup, preds, bad, A.A_WORDS, "kappa", N_A)
                    if version == "v1":
                        c["B"] = A.dev_cell(y, lookup, preds, bad, A.B_WORDS, "share", N_B)
                else:
                    c["A_descriptive"] = A.paired(y, lookup, preds, "kappa")
                    if version == "v1":
                        c["B_descriptive"] = A.paired(y, y, preds, "share")
            if split == "labelled":
                c["vs_person"] = A.person_figures(ids, preds, person)
            m["cells"][f"{version}|{split}"] = c
        out["models"][model] = m
    spend = _jsonl(spend_path or K.SPEND)
    out["cost"] = costs(spend)
    out["account"] = account(events)
    out["summary"] = summary(out)
    out["table"] = table_markdown(out)
    return out


def summary(out: dict) -> str:
    """Amendment 1's sentence for this branch's comparison set (section 10), then the registered sentences."""
    counted, excluded = [], []
    open_ = [f"{t} ({out['models'][t]['status']})" for t in out["order"] if not out["models"][t]["closed"]]
    for model in out["order"]:
        m = out["models"][model]
        if not m["compared"] or not m["closed"]:
            continue
        cells, why = m["cells"], None
        if m["status"] != "complete":
            why = m["status"]
        elif any(c.get("status") != "complete" for c in cells.values()):
            why = "; ".join(f"{k}: {c['status']}" for k, c in cells.items() if c.get("status") != "complete")
        elif any(c.get(x, {}).get("format_sensitive") for c in cells.values() for x in ("A", "B")):
            why = "format-sensitive"
        (excluded.append(f"{model} ({why})") if why else counted.append(model))
    m_ = len(counted)
    k = sum(out["models"][t]["cells"]["v1|dev"]["B"]["label"] == A.B_WORDS[0] for t in counted)
    a_labels = [out["models"][t]["cells"][f"{v}|dev"]["A"]["label"] for t in counted for v in ("v1", "v2")]
    j, a = a_labels.count(A.A_WORDS[1]), a_labels.count(A.A_WORDS[0])
    if open_:
        s = "Cloud branch, on DEV: not finished, so no count is given; not closed: " + "; ".join(open_) + "."
        a = 0
    else:
        s = (f"Cloud branch, on DEV: judge v1 paid more often than the companies did for {k} of {m_} models; across "
             f"the {m_} models and both prompts, {j} of {2 * m_} comparisons with the issuer lookup were below it, {a} "
             f"were above it and the rest were not distinguishable ({', '.join(counted) or 'none'}).")
    if a:
        s += (" This is one run of each model as OpenRouter and the pinned provider served it on the dates of the run; "
              "the models may have seen these complaints and their outcomes in training (see Amendment 3, section "
              "11).")
    s += (" The design compares each model with the issuer lookup and with the record, not models with one another; "
          "it does not show that the results are independent of the model.")
    if excluded:
        s += " Not counted: " + "; ".join(excluded) + "."
    rows = [f"{t} ({out['models'][t]['row_note']}; {out['models'][t]['status']})" for t in out["order"]
            if not out["models"][t]["compared"] and out["models"][t]["closed"]]
    if rows:
        s += " Reference rows, descriptive only: " + "; ".join(rows) + "."
    s += " " + PERSON_NOTE
    c = out.get("cost")
    if c:
        s += (" Cost: " + "; ".join(f"{t} {c[t]['usd']:.4f} USD ({c[t]['answers']} answers, "
                                    f"{c[t]['charged_without_answer']} charged without an answer)"
                                    for t in K.MODELS if c[t]["answers"] or c[t]["usd"])
              + f"; {c['total_usd']:.4f} USD in all, under a cap of {c['cap_usd']:.0f} USD"
              + (" (free models cost nothing)." if any(c[t]["answers"] for t in K.FREE_QUEUE) else "."))
    acc = out.get("account") or {}
    if acc:
        s += (f" Credit remaining by the API: {acc['first']['remaining_credits']} USD before the first run, "
              f"{acc['last']['remaining_credits']} USD after the last.")
    return s


def _iv(k):
    return f"{k['value']:.3f} [{k['lo']:.3f}, {k['hi']:.3f}]" if k and k.get("value") is not None else "-"


def table_markdown(out: dict) -> str:
    lines = ["| Model | Arm | Cell | Answered | Malformed | Pays | Kappa vs record | Label against the issuer lookup | "
             "Payout label | Kappa vs the person (decided) | Reasoning tokens (median / max) | Other finishes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in out["order"]:
        m = out["models"][model]
        name = f"`{model}`" + (f" ({m['row_note']})" if m["row_note"] else "") + (
            f" ({m['status']})" if m["status"] != "complete" else "")
        if not any(c["answered"] for c in m["cells"].values()):
            lines.append(f"| {name} | {m['arm']} | - | 0 | - | - | - | - | - | - | - | - |")
            continue
        for cell, c in m["cells"].items():
            a = c.get("A", {}).get("label") or ("descriptive only" if "A_descriptive" in c else "-")
            b = c.get("B", {}).get("label") or ("descriptive only" if "B_descriptive" in c else "-")
            person = _iv(c["vs_person"]["kappa_decided"]) if "vs_person" in c else "-"
            rt = c.get("reasoning_tokens")
            reasoning = f"{rt['median']:g} / {rt['max']}" if rt else "-"
            other = ", ".join(f"{k} {v}" for k, v in c.get("finish_reasons", {}).items() if k != "stop") or "-"
            lines.append(f"| {name} | {m['arm']} | {cell} | {c['answered']} of {c['n']} | {c['malformed']} | "
                         f"{c.get('pays', c.get('pays_answered', '-'))} | {_iv(c.get('kappa_vs_record'))} | {a} | {b} | "
                         f"{person} | {reasoning} | "
                         f"{other} |")
    return "\n".join(lines)


def main() -> None:
    out = analyse()
    K.DIR.mkdir(parents=True, exist_ok=True)
    (K.DIR / "analysis.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(out["table"] + "\n\n" + out["summary"])


if __name__ == "__main__":
    main()
