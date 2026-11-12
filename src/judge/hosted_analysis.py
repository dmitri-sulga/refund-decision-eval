"""Figures and labels of the hosted arm (docs/analysis_plan_models.md, Amendment 2), in its own table.

Written before any hosted request. Applies the local extension's computations unchanged (models_analysis: the
malformed rule, the paired intervals, the three readings, the person's figures). Only the models that ran with
reasoning off form the comparison set (Amendment 2, section 3): claude-haiku-5-5 and claude-sonnet-5-5, 4 comparisons
with the issuer lookup and 2 payout comparisons. claude-opus-5-5, whose reasoning cannot be switched off, is a
descriptive reference row: its figures are given, it gets no label and it is not counted. Reads the arm's cache, run
log and spend journal, the study's cache (for qwen3.8:27b) and the committed data. Calls no model.

python -m judge.hosted_analysis            # writes results/models/hosted/analysis.json and prints table and text
"""
from __future__ import annotations

import json
import statistics

import numpy as np

from . import config as C
from . import hosted as H
from . import metrics as M
from . import models_analysis as A
from . import models as XM
from .models import CELLS, PLAN_FILE, REFERENCE, _jsonl

N_A = 2 * len(H.COMPARED)      # 4 comparisons with the issuer lookup (models x prompts), fixed before any request
N_B = len(H.COMPARED)          # 2 payout comparisons, fixed before any request
REFERENCE_ROW = "reasoning on, cannot be disabled; 4096 max_tokens; not part of the comparisons"
MOVED_ROW = "reasoning on (see Amendment 2, section 3); not part of the comparisons"
THINKING_BLOCKS = ("thinking", "redacted_thinking")


def plan_paragraph(label: str) -> str:
    """A paragraph of Amendment 2, printed exactly as the plan has it (one source): from "**label**" to the next blank
    line, with the bold markers removed and the lines joined."""
    text = PLAN_FILE.read_text(encoding="utf-8")
    a2 = text.index("## Amendment 2")
    start = text.index(f"**{label}**", a2)
    end = text.index("\n\n", start)
    return " ".join(text[start:end].split()).replace(f"**{label}**", label, 1)


DISCLOSURE = plan_paragraph("Same vendor.")
OPUS_SENTENCE = ("claude-opus-5-5 ran with adaptive thinking, which it does not allow to be turned off, at effort low "
                 "and max_tokens 4096. Its thinking tokens are given per cell in the table. It is a reference row, not "
                 "part of the comparisons.")
PERSON_NOTE = ("The one person who labelled the sample agreed with the record at kappa 0.202 [-0.054, 0.456] on his 56 "
               "decided labels, so neither the record nor this person is a gold standard. Each model's closeness to "
               "the record and to this person is given with its interval; the models are not ranked, and no model is "
               "called the best.")


def reasoning_off(model: str, answers: list[dict]) -> bool:
    """Section 3: in the comparison set only if registered there and no answer reports thinking tokens above 0 or a
    thinking or redacted_thinking block (a single token is enough: the rule is mechanical and fixed in advance)."""
    return (model in H.COMPARED
            and not any((r.get("thinking_tokens") or 0) > 0 or set(THINKING_BLOCKS) & set(r.get("block_types") or [])
                        for r in answers))


def opus_note(out: dict) -> str:
    """The registered sentence on claude-opus-5-5, printed only when it has answers; the clause on the compared
    models names only those that ran in the comparison set."""
    if not out["models"]["claude-opus-5-5"]["cells"] or not any(
            c["answered"] for c in out["models"]["claude-opus-5-5"]["cells"].values()):
        return ""
    ran = [t for t in H.COMPARED if out["models"][t]["compared"]
           and any(c["answered"] for c in out["models"][t]["cells"].values())]
    s = OPUS_SENTENCE
    if ran:
        s += (f" {' and '.join(ran)} ran without up-front thinking and max_tokens 512, as the local models did.")
    return s


def costs(spend: list[dict]) -> dict:
    """Per model: answers, charges without an answer, tokens and USD, from the spend journal. A reservation that no
    answer, charge or release settled (a crash) is counted at its worst case. An answer billed but lost before it was
    cached (a crash) and then resent counts as a charge without an answer."""
    out = {}
    open_ = H.spend_state(spend)["open"]
    for t in H.MODELS:
        mine = [e for e in spend if e.get("model") == t and e.get("event") in ("answer", "charged")]
        ans = list({e["key"]: e for e in mine if e["event"] == "answer"}.values())       # the last per request
        reserved = [e for e in open_ if e.get("model") == t]
        out[t] = {"answers": len(ans), "charged_without_answer": len(mine) - len(ans),
                  "open_reservations": len(reserved),
                  **{f: sum(e.get(f) or 0 for e in ans) for f in ("input_tokens", "output_tokens", "thinking_tokens")},
                  "usd": round(sum(e["cost_usd"] for e in mine) + sum(e["cost_usd"] for e in reserved), 4)}
    out["total_usd"] = round(sum(out[t]["usd"] for t in H.MODELS), 4)
    out["cap_usd"] = H.CAP_USD
    return out


def model_status(model: str, closed: dict, answered: int, n: int, budget_stop: bool, run_log: list[dict]) -> tuple:
    """(status in plain words, closed by a registered condition). Only the code closes a model (section 9), so every
    closed model is closed by a registered condition; a model that is not closed makes the summary not final."""
    e = closed.get(model)
    if e and e["event"] == "model_finished":
        return "complete", True
    if e:
        return e["reason"], True
    if budget_stop and not answered:
        return "not run: no request of this arm is sent after a budget stop", True
    stops = [x for x in run_log if x.get("model") == model and x.get("event") in ("run_started", "run_stopped")]
    last = (f"; its last run stopped: {H.VERDICT_WORDS.get(stops[-1].get('verdict'), 'an interruption')}"
            if stops and stops[-1]["event"] == "run_stopped" else "")
    return (f"not closed ({answered} of {n} answered{last})" if answered or stops else "not started"), False


def resends_and_losses(journal: list[dict], spend: list[dict], run_log: list[dict]) -> dict:
    """Section 9: every resend, and every charge without an answer with its recorded cause, per model."""
    out = {t: {"resent": [], "losses": []} for t in H.MODELS}
    for e in journal:
        t = XM.split_key(e["key"])[0]
        if e.get("kind") == "resend" and t in out:
            out[t]["resent"].append(e["key"])
    stops: dict[str, list[str]] = {}
    for e in run_log:
        if e.get("event") == "run_stopped" and e.get("verdict") == "lost":
            stops.setdefault(e["key"], []).append(str(e.get("reason") or "").rstrip(": ") or "lost")
    for e in spend:
        if e.get("event") != "charged" or e.get("model") not in out:
            continue
        note = str(e.get("note") or "")
        if note.startswith("answer served by") or note.endswith("(crash)"):
            cause = note
        else:
            cause = (stops.get(e["key"]) or [note]).pop(0)
        out[e["model"]]["losses"].append(f"{e['key']}: {cause[:100]}")
    return out


def analyse(cache_path=None, study_cache_path=None, records=None, run_log=None, spend_path=None,
            journal_path=None) -> dict:
    from .baselines import issuer_rate
    from .data import load_records, load_splits
    from .label import labels_path, load_labels

    records = records if records is not None else load_records()
    splits = load_splits()
    sets, _ = XM.item_sets()
    cache = {r["key"]: r for r in _jsonl(cache_path or H.CACHE_FILE)}
    study_cache = {r["key"]: r for r in _jsonl(study_cache_path or C.CACHE_FILE)}
    events = _jsonl(run_log or H.RUN_LOG)
    closed = {e["model"]: e for e in events if e.get("event") in ("closed_incomplete", "model_finished")}
    budget_stop = any(e.get("event") == "budget_stop" for e in events)
    f, base, _ = issuer_rate(splits["train"], records)
    dev, lab = sets["dev"], sets["labelled"]
    y = np.array([records[i]["label"] for i in dev])
    lookup = np.array([int(f(records[i]) > base) for i in dev])
    person = load_labels(labels_path("owner"))
    out = {"arm": "hosted (Amendment 2)", "order": list(H.MODELS), "reference": REFERENCE[0],
           "record_pays_dev": int(y.sum()), "n_dev": len(dev), "lookup_pays_dev": int(lookup.sum()),
           "comparisons": {"A": N_A, "B": N_B}, "models": {}}
    ref = {}
    for version, split in CELLS:
        ids = dev if split == "dev" else lab
        ans = A.cell_answers(REFERENCE[0], version, ids, cache, study_cache)
        ref[(version, split)] = np.array([ans[i][0] for i in ids])
    for model in H.MODELS:
        mine = [r for k, r in cache.items() if k.startswith(model + "|")]
        n_all = sum(len(dev if s == "dev" else lab) for _, s in CELLS)
        status, legit = model_status(model, closed, len(mine), n_all, budget_stop, events)
        compared = reasoning_off(model, mine)
        m = {"status": status, "closed": legit, "compared": compared,
             "row_note": None if compared else REFERENCE_ROW if model in H.REFERENCE_ONLY else MOVED_ROW,
             "cells": {}}
        for version, split in CELLS:
            ids = dev if split == "dev" else lab
            ans = A.cell_answers(model, version, ids, cache, study_cache)
            c = {"n": len(ids), "answered": len(ans), "malformed": sum(v[1] for v in ans.values())}
            stops = [v[2].get("stop_reason") for v in ans.values()]
            c["stop_reasons"] = {s: stops.count(s) for s in sorted(set(stops), key=str)}
            c["routed_elsewhere"] = sum(1 for v in ans.values()
                                        if v[2].get("inference_geo") not in (H.INFERENCE_GEO, None))
            c["refusal_categories"] = sorted({(v[2].get("stop_details") or {}).get("category") or "none"
                                              for v in ans.values() if v[2].get("stop_reason") == "refusal"})
            walls = [v[2]["wall_s"] for v in ans.values() if v[2].get("wall_s") is not None]
            if walls:
                c["seconds_per_call"] = {"median": statistics.median(walls), "mean": statistics.fmean(walls)}
            thinking = [v[2].get("thinking_tokens") or 0 for v in ans.values()]
            if thinking:
                c["thinking_tokens"] = {"median": statistics.median(thinking), "max": max(thinking),
                                        "total": sum(thinking)}
            if len(ans) < len(ids):
                c["status"] = f"incomplete ({len(ans)} of {len(ids)} answered)"
                c["pays_answered"] = sum(v[0] for v in ans.values())
                m["cells"][f"{version}|{split}"] = c
                continue
            preds = np.array([ans[i][0] for i in ids])
            bad = np.array([ans[i][1] for i in ids])
            c["pays"] = int(preds.sum())
            c["kappa_with_reference"] = M.cohen_kappa(ref[(version, split)], preds)
            c["status"] = "not interpretable (format)" if c["malformed"] > A.FORMAT_LIMIT[split] else "complete"
            yy = y if split == "dev" else np.array([records[i]["label"] for i in ids])
            c["kappa_vs_record"] = M.bootstrap(yy, preds)["kappa"]
            if split == "dev" and c["status"] == "complete":
                if compared:
                    c["A"] = A.dev_cell(y, lookup, preds, bad, A.A_WORDS, "kappa", N_A)
                    if version == "v1":
                        c["B"] = A.dev_cell(y, lookup, preds, bad, A.B_WORDS, "share", N_B)
                else:                             # descriptive: the 95% intervals, no label, not counted
                    c["A_descriptive"] = A.paired(y, lookup, preds, "kappa")
                    if version == "v1":
                        c["B_descriptive"] = A.paired(y, y, preds, "share")
            if split == "labelled":
                c["vs_person"] = A.person_figures(ids, preds, person)
            m["cells"][f"{version}|{split}"] = c
        out["models"][model] = m
    spend = _jsonl(spend_path or H.SPEND)
    out["cost"] = costs(spend)
    out["resends_and_losses"] = resends_and_losses(_jsonl(journal_path or H.JOURNAL), spend, events)
    out["summary"] = summary(out)
    out["table"] = table_markdown(out)
    return out


def summary(out: dict) -> str:
    """The sentence of Amendment 1, section 3, for this arm's comparison set, then the registered sentences."""
    counted, excluded = [], []
    open_ = [f"{t} ({out['models'][t]['status'].removeprefix('not closed (').removesuffix(')')})"
             if out["models"][t]["status"].startswith("not closed (") else f"{t} ({out['models'][t]['status']})"
             for t in out["order"] if not out["models"][t]["closed"]]
    for model in out["order"]:
        m = out["models"][model]
        cells = m["cells"]
        why = None
        if not m["compared"] or not m["closed"]:
            continue
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
    if open_:                                     # a model not closed: no count is given (section 10)
        s = "Hosted arm, on DEV: not finished, so no count is given; not closed: " + "; ".join(open_) + "."
        a = 0
    else:
        s = (f"Hosted arm, on DEV: judge v1 paid more often than the companies did for {k} of {m_} models; across "
             f"the {m_} models and both prompts, {j} of {2 * m_} comparisons with the issuer lookup were below it, {a} "
             f"were above it and the rest were not distinguishable ({', '.join(counted) or 'none'}).")
    if a:
        s += (" This is one run of each model as the provider served it on the dates of the run; the models may have "
              "seen these complaints and their outcomes in training (see Amendment 2, section 14).")
    s += (" The design compares each model with the issuer lookup and with the record, not models with one another; "
          "it does not show that the results are independent of the model.")
    if excluded:
        s += " Not counted: " + "; ".join(excluded) + "."
    rows = [f"{t} ({out['models'][t]['row_note']}; {out['models'][t]['status']})" for t in out["order"]
            if not out["models"][t]["compared"] and out["models"][t]["closed"]]
    if rows:
        s += " Reference rows, descriptive only: " + "; ".join(rows) + "."
    note = opus_note(out)
    s += (" " + note if note else "") + " " + PERSON_NOTE + " " + DISCLOSURE
    c = out.get("cost")
    if c:
        s += (" Cost: " + "; ".join(
            f"{t} {c[t]['usd']:.2f} USD ({c[t]['answers']} answers, {c[t]['charged_without_answer']} charged without "
            f"an answer; {c[t]['input_tokens']} input, {c[t]['output_tokens']} output and {c[t]['thinking_tokens']} "
            "thinking tokens)" for t in H.MODELS) +
              f"; {c['total_usd']:.2f} USD in all, under a cap of {c['cap_usd']:.0f} USD.")
        if any(c[t]["open_reservations"] for t in H.MODELS):
            s += (" This includes " + ", ".join(f"{c[t]['open_reservations']} of {t}" for t in H.MODELS
                                                if c[t]["open_reservations"]) +
                  " open reservation(s), at their worst case: requests whose outcome a crash left unknown.")
    rl = out.get("resends_and_losses") or {}
    parts = [f"{t}: {len(v['resent'])} resent" + (f" ({', '.join(v['resent'])})" if v["resent"] else "") +
             f", {len(v['losses'])} charged without an answer" + (f" ({'; '.join(v['losses'])})" if v["losses"] else "")
             for t, v in rl.items() if v["resent"] or v["losses"]]
    if rl:
        s += (" Resends and losses: " + "; ".join(parts) + "." if parts
              else " No request was resent or charged without an answer.")
    return s


def _iv(k: dict) -> str:
    return f"{k['value']:.3f} [{k['lo']:.3f}, {k['hi']:.3f}]" if k and k.get("value") is not None else "-"


def _stops(c: dict) -> str:
    """Stop reasons other than end_turn, with the refusal categories."""
    other = {k: v for k, v in c.get("stop_reasons", {}).items() if k != "end_turn"}
    text = ", ".join(f"{k} {v}" for k, v in other.items())
    cats = [x for x in c.get("refusal_categories", []) if x != "none"]
    return (text + (f" ({', '.join(cats)})" if cats else "")) or "-"


def table_markdown(out: dict) -> str:
    """The hosted table, in the order of section 2, with each reference row labelled where it is printed and the
    per-cell figures section 10 adds for this arm. The disclosure and the Opus sentence are in the summary, once."""
    lines = ["| Model | Cell | Answered | Malformed | Pays | Kappa vs record | Label against the issuer lookup | "
             "Payout label | Kappa vs the person (decided) | Thinking tokens (median / max) | Routing not global | "
             "Other stops |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in out["order"]:
        m = out["models"][model]
        name = f"`{model}`" + (f" ({m['row_note']})" if m["row_note"] else "") + (
            f" ({m['status']})" if m["status"] != "complete" else "")
        if not m["cells"] or not any(c["answered"] for c in m["cells"].values()):
            lines.append(f"| {name} | - | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        for cell, c in m["cells"].items():
            a = c.get("A", {}).get("label") or ("descriptive only" if "A_descriptive" in c else "-")
            b = c.get("B", {}).get("label") or ("descriptive only" if "B_descriptive" in c else "-")
            person = _iv(c["vs_person"]["kappa_decided"]) if "vs_person" in c else "-"
            th = c.get("thinking_tokens")
            thinking = f"{th['median']:g} / {th['max']}" if th else "-"
            pays = c.get("pays", c.get("pays_answered", "-"))
            lines.append(f"| {name} | {cell} | {c['answered']} of {c['n']} | {c['malformed']} | {pays} | "
                         f"{_iv(c.get('kappa_vs_record'))} | {a} | {b} | {person} | {thinking} | "
                         f"{c['routed_elsewhere']} | {_stops(c)} |")
    return "\n".join(lines)


def main() -> None:
    out = analyse()
    H.DIR.mkdir(parents=True, exist_ok=True)
    (H.DIR / "analysis.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(out["table"] + "\n\n" + out["summary"])


if __name__ == "__main__":
    main()
