"""Compute every reported number from data + cached responses, then render the
README tables, results/metrics.json and results/report.md.

python -m judge.report           # regenerate everything (offline, from cache)
python -m judge.report --check   # fail (exit 1) if any committed number differs
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys

from . import config as C
from .analysis import compute

README = C.ROOT / "README.md"
METRICS_FILE = C.RESULTS / "metrics.json"
REPORT_FILE = C.RESULTS / "report.md"
BLOCK_RE = re.compile(r"(<!-- BEGIN (\w+) -->\n)(.*?)(<!-- END \2 -->)", re.S)


def pct(x, d=1):
    return "n/a" if x is None or x != x else f"{100 * x:.{d}f}%"


def num(x, d=3):
    if x is None or x != x:
        return "n/a"
    out = f"{x:.{d}f}"
    return out[1:] if out.startswith("-") and float(out) == 0 else out


def ci(m, f=num):
    return f"{f(m['value'])} [{f(m['lo'])}, {f(m['hi'])}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


PENDING = "_Not run yet._"


# ----------------------------------------------------------------- blocks
def b_settings(m):
    s = m["settings"]
    env = m.get("environment") or {}
    rows = [
        ("Model", f"`{s['model']}` via Ollama (local)"),
        ("Ollama version / model digest", f"{env.get('ollama_version', 'pending')} / "
                                          f"`{env.get('model_digest', 'pending')}`"),
        ("Decoding", f"temperature {s['options']['temperature']}, seed {s['options']['seed']}, "
                     f"thinking disabled, JSON schema enforced, num_ctx {s['options']['num_ctx']}"),
        ("Narrative cap", f"{s['max_narrative_chars']} characters (same text for every model)"),
        ("Split sizes (DEBUG / DEV / TEST / TRAIN)", " / ".join(str(s['sizes'][k]) for k in
                                                               ("debug", "dev", "test", "train"))),
        ("Split seed", s["split_seed"]),
        ("Bootstrap", f"{s['bootstrap_b']} resamples, seed {s['bootstrap_seed']}, percentile 95% CI"),
        ("Automation target", f"agreement >= {s['target_agreement']:.2f} on automated DEV cases, "
                              f"at least {s['min_automated']} automated"),
        ("Gaming subset", f"{s['n_gaming']} DEV complaints x {s['n_attacks']} rewrites, seed {s['gaming_seed']}"
                          f" (plus {s['n_controls']} control sentence, added later)"),
        ("Near-duplicate rule", f"word-trigram TF-IDF cosine >= {s['near_dup_cosine']} (collapsed); "
                                f"ceiling proxy uses same-company pairs >= {s['ceiling_pair_cosine']}"),
    ]
    return table(("Setting", "Value"), rows)


def b_data(m):
    d = m["data"]
    f = d["funnel"]
    rows = [
        ("Rows in the archived copy of the bulk export", f"{d['mirror_rows_total']:,}"),
        ("Credit card / purchase-on-statement complaints received in the window", f"{d['slice_rows']:,}"),
        ("... with a consumer narrative", f"{f['slice_with_narrative']:,}"),
        ("... dropped: status not mapped (in progress / untimely)", f"{f['dropped_status_not_mapped']:,}"),
        ("... dropped: metadata differs from the live API", f"{f['dropped_metadata_mismatch_vs_live_api']:,}"),
        ("... dropped: not found in the live API", f"{f.get('dropped_not_in_live_api', 0):,}"),
        ("... removed as near-duplicates (re-filed or templated text)", f"{f['records_removed_as_near_duplicates']:,}"),
        ("Sampling frame", f"{f['sampling_frame']:,}"),
    ]
    st = d["status_counts"]
    srows = [(k, f"{v:,}", d["label_map"].get(k, "dropped")) for k, v in
             sorted(st.items(), key=lambda kv: -kv[1])]
    return (table(("Step", "Complaints"), rows) + "\n\n" +
            table(("Recorded company response", "Complaints with narrative", "Label"), srows))


def b_base_rates(m):
    rows = [(k.upper(), f"{v['n']:,}", pct(v["base_rate"])) for k, v in m["base_rates"].items()]
    return table(("Split", "Complaints", "Share closed with monetary relief"), rows)


LABELS = {
    "majority": "Majority class (always no relief)",
    "keyword_rule": "Keyword rule (chosen on TRAIN)",
    "company_rule": "Company name only (company's TRAIN relief rate above one half)",
    "tfidf_lr": "TF-IDF + logistic regression (trained on TRAIN)",
    "v1": "LLM judge v1 (plain rubric)",
    "v2": "LLM judge v2 (facts first, then decision)",
    "guarded": "Guarded pipeline (v2 facts -> fact model; filter routes to human)",
}


def results_table(res: dict) -> str:
    rows = []
    for name, r in res.items():
        if name not in LABELS:
            continue
        rows.append((LABELS[name], r["n"], ci(r["accuracy"]), ci(r["kappa"]),
                     num(r["precision_1"]["value"]), num(r["recall_1"]["value"]),
                     num(r["precision_0"]["value"]), num(r["recall_0"]["value"]), r["invalid"]))
    return table(("System", "n", "Accuracy [95% CI]", "Kappa [95% CI]", "Precision relief",
                  "Recall relief", "Precision no-relief", "Recall no-relief", "Invalid"), rows)


def diff_sentence(res: dict, a: str, b: str, where: str) -> str | None:
    x = res.get(f"{b}_minus_{a}")
    if not x:
        return None
    t = x["transitions"]
    return (f"{LABELS[b]} minus {LABELS[a]} on the same {where} complaints (paired bootstrap): kappa "
            f"{ci(x['kappa'], lambda v: f'{v:+.3f}')}, accuracy {ci(x['accuracy'], lambda v: f'{v:+.3f}')}. "
            f"The second system agrees with the record where the first did not on {t['only_second_agrees']} "
            f"complaints ({t['gained_no_relief_cases']} closed without relief, {t['gained_relief_cases']} with) "
            f"and loses {t['only_first_agrees']} the first had right ({t['lost_relief_cases']} closed with relief, "
            f"{t['lost_no_relief_cases']} without).")


def confusion_table(res: dict) -> str:
    rows = []
    for name, r in res.items():
        if name in LABELS:
            c = r["confusion"]
            rows.append((LABELS[name], c["relief_relief"], c["relief_none"], c["none_relief"], c["none_none"]))
    return table(("System", "Recorded relief, system relief", "Recorded relief, system none",
                  "Recorded none, system relief", "Recorded none, system none"), rows)


def b_dev_results(m):
    d = m["dev"]
    out = [results_table(d), "\nConfusion counts:\n", confusion_table(d)]
    for a, b in (("v1", "v2"), ("v2", "guarded"), ("tfidf_lr", "v1"), ("tfidf_lr", "v2")):
        sent = diff_sentence(d, a, b, "DEV")
        if sent:
            out.append("\n" + sent)
    out.append("\nNo acceptance rule for v2 was written before this DEV comparison; the rule for the "
               "held-out comparison is in the frozen file.")
    k = m["keyword_rule"]
    out.append(f"\nKeyword rule: of {len(k['candidates'])} candidate rules scored on TRAIN, the best was "
               f"\"{k['chosen']['rule']}\" (TRAIN kappa {num(k['chosen']['train_kappa'])}, predicts relief "
               f"for {pct(k['chosen']['train_predicted_relief'])} of TRAIN).")
    lr = m["tfidf_lr"]
    best = next(r for r in lr["cv_kappa"] if r["C"] == lr["chosen"]["C"]
                and r["class_weight"] == lr["chosen"]["class_weight"])
    out.append(f"\nLogistic regression settings chosen by 5-fold CV on TRAIN: C = {lr['chosen']['C']}, "
               f"class weighting = {lr['chosen']['class_weight']} (CV kappa {num(best['cv_kappa'])}). "
               "DEV kappa as a function of the number of TRAIN labels:\n")
    out.append(table(("TRAIN labels", "DEV kappa", "DEV accuracy"),
                     [(f"{r['n_train']:,}", num(r["kappa"]), num(r["accuracy"]))
                      for r in lr["learning_curve"]]))
    return "\n".join(out)


def b_date_halves(m):
    d = m["date_halves"]
    r = d["ranges"]
    names = list(d["earlier"]["systems"])
    rows = [(LABELS[n],) + tuple(ci(d[h]["systems"][n]["kappa"]) for h in ("earlier", "later")) for n in names]
    return "\n".join([
        "The split is random within the window, not by date. Dates received: " +
        "; ".join(f"{k.upper()} {r[k]['from']} to {r[k]['to']} (median {r[k]['median']})" for k in r) +
        ". Kappa on the earlier and the later half of DEV:\n",
        table(("System", f"Earlier half ({d['earlier']['from']} to {d['earlier']['to']}, n={d['earlier']['n']}, "
                         f"{pct(d['earlier']['base_rate'])} relief)",
               f"Later half ({d['later']['from']} to {d['later']['to']}, n={d['later']['n']}, "
               f"{pct(d['later']['base_rate'])} relief)"), rows)])


def time_per_1000(m, name):
    c = m["cost"]
    if name == "majority":
        return "none"
    if name == "keyword_rule" and "tfidf_lr" in c:
        return f"{c['tfidf_lr']['keyword_rule_ms_per_item']:.2f} s (CPU)"
    if name == "tfidf_lr" and "tfidf_lr" in c:
        return f"{c['tfidf_lr']['predict_ms_per_item_one_by_one']:.1f} s (CPU)"
    v = "v2" if name == "guarded" else name
    if v in c:
        return f"{c[v]['hours_per_1000']:.1f} h (one laptop GPU)"
    return "n/a"


def b_summary(m):
    b = m["base_rates"]
    t = m.get("test")
    head = (f"DEV: {b['dev']['n']} complaints, {pct(b['dev']['base_rate'])} closed with monetary relief. "
            f"Held-out TEST: {b['test']['n']} complaints, {pct(b['test']['base_rate'])}"
            + ("" if t else ", not yet run") +
            f". TRAIN (classic baselines only): {b['train']['n']:,}, {pct(b['train']['base_rate'])}. "
            "Kappa is Cohen's kappa against the recorded company decision; intervals are 95% bootstrap.\n")
    rows = []
    for name in ("majority", "keyword_rule", "tfidf_lr", "v1", "v2", "guarded"):
        if name not in m["dev"]:
            continue
        r = m["dev"][name]
        op = (m["automation_dev"].get(name) or {}).get("token")
        if name in ("majority", "keyword_rule"):
            auto = "no confidence signal"
        elif not op or op["at_threshold"] is None:
            auto = "none reaches the target"
        else:
            at = op["at_threshold"]
            auto = (f"{pct(at['coverage']['value'])} automated, {ci(at['agreement'], pct)} agreement "
                    f"({pct(at['majority_agreement_on_auto'])} for always-no-relief on the same cases)")
        held = ci(t["results"][name]["kappa"]) if t and name in t["results"] else "not run"
        rows.append((LABELS[name], ci(r["kappa"]), pct(r["accuracy"]["value"]), auto, held, time_per_1000(m, name)))
    people = [(who, r) for who, r in m["human"]["annotators"].items() if "person_vs_recorded" in r["all"]]
    for who, r in people:
        f = r["all"]
        rows.append((f"Person, blind (one annotator, {f['decided']} decided DEV cases)",
                     ci(f["person_vs_recorded"]["kappa"]), pct(f["person_vs_recorded"]["accuracy"]["value"]),
                     "n/a", "n/a", f"{r['median_seconds'] * 1000 / 3600:.1f} h (median labelling time)"))
    if not people:
        rows.append(("Person, blind (one annotator)", "not labelled yet", "", "n/a", "n/a", ""))
    return head + "\n" + table(("System", "DEV kappa [95% CI]", "DEV accuracy",
                                "Automation at the DEV operating point (target in settings)",
                                "Held-out kappa [95% CI]", "Time per 1,000 decisions"), rows)


def b_inputs(m):
    r = m["dev"]["company_rule"]
    st = m["settings"]
    return (f"Narratives are cut at {st['max_narrative_chars']:,} characters for every system. "
            f"Model: `{st['model']}`, temperature {st['options']['temperature']}, seed {st['options']['seed']}. "
            f"Company name alone, which no other system is given, reaches kappa {ci(r['kappa'])} on DEV "
            f"(accuracy {pct(r['accuracy']['value'])}).")


def b_error_groups(m):
    e = m.get("error_analysis")
    if not e:
        return PENDING
    rows = []
    for g in e["groups"]:
        ex = g["example"]
        rows.append((g["label"], g["n"], g["read_individually"],
                     (f"\"{ex['quote']}\" ({ex['complaint_id']}" + (f": {ex['note']})" if ex["note"] else ")"))
                     if ex else ""))
    rev = [r["error_review"] for r in m["human"]["annotators"].values() if r["error_review"]["reviewed"]]
    who = ("The categories and this three-way reading were drafted by an AI coding assistant that read each "
           "disagreement during the analysis (an assistant, not the judged model). " +
           (f"The author has since reviewed {rev[0]['reviewed']} of the {rev[0]['total']} individually sorted "
            f"cases himself and confirmed {rev[0]['confirmed']} of the drafted readings (changes are listed in the "
            "full report)." if rev else
            "The author has not yet confirmed them; he does that for the individually sorted cases after his "
            "blind labelling, and the counts will be added here."))
    return (f"All {e['n_disagreements']} DEV disagreements of judge v1 (of {e['n_dev']} complaints). {who}\n\n" +
            table(("Reading", "Cases", "... assigned by reading the case (the rest follow from their category)",
                   "Example"), rows))


def verdicts(m) -> list[str]:
    """Plain-language reading of the DEV table, derived from the intervals."""
    d = m["dev"]
    ranked = sorted((n for n in d if n in LABELS and n != "company_rule"),
                    key=lambda n: -d[n]["kappa"]["value"])
    out = [f"Highest DEV kappa: {LABELS[ranked[0]]}."]
    for v in ("v1", "v2"):
        x = d.get(f"{v}_minus_tfidf_lr")
        if x:
            out.append(f"{LABELS[v]} is " + ("below the TF-IDF baseline (the paired interval excludes zero)."
                                              if x["kappa"]["hi"] < 0 else
                                              "above the TF-IDF baseline (the paired interval excludes zero)."
                                              if x["kappa"]["lo"] > 0 else
                                              "not distinguishable from the TF-IDF baseline (the paired interval includes zero)."))
    x = d.get("v2_minus_v1")
    if x:
        k = x["kappa"]
        out.append(("The prompt change from v1 to v2 raised kappa." if k["lo"] > 0 else
                    "The prompt change from v1 to v2 lowered kappa." if k["hi"] < 0 else
                    "The prompt change from v1 to v2 did not move kappa outside its interval: no improvement "
                    "is demonstrated on DEV.") +
                   f" Accuracy rose because v2 predicts relief for {pct(d['v2']['predicted_positive_rate'])} of "
                   f"complaints where v1 predicted it for {pct(d['v1']['predicted_positive_rate'])}; "
                   f"recall of relief fell from {num(d['v1']['recall_1']['value'], 2)} to "
                   f"{num(d['v2']['recall_1']['value'], 2)}.")
    return out


def b_v1v2(m):
    return "\n\n".join([" ".join(verdicts(m))] +
                       [x for x in (diff_sentence(m["dev"], "v1", "v2", "DEV"),
                                    diff_sentence(m["dev"], "tfidf_lr", "v2", "DEV")) if x])


def b_retest_summary(m):
    rt = m.get("retest")
    if not rt:
        return PENDING
    return " ".join(
        f"Asked twice with identical settings, {LABELS[v]} gave the same decision in {r['same_decision']}/{r['n']} = "
        f"{pct(r['same_decision_rate'])} [{pct(r['same_decision_ci'][0])}, {pct(r['same_decision_ci'][1])}] of DEV "
        f"requests (largest change in P(relief) {num(r['shift']['max'], 2)}; kappa against the record "
        f"{num(r['kappa_cached'])} from the cached answers and {num(r['kappa_replay'])} from the replayed ones "
        "on the same requests)." for v, r in rt.items())


def b_error_analysis(m):
    e = m.get("error_analysis")
    if not e:
        return PENDING
    bd = e["by_direction"]
    head = (f"Judge v1 disagreed with the recorded outcome on {e['n_disagreements']} of "
            f"{e['n_dev']} DEV complaints: {bd.get('judge relief, company none', 0)} where it granted "
            f"relief and the company did not, {bd.get('judge none, company relief', 0)} the other way "
            "round. Every disagreement has exactly one primary category "
            "(`results/error_analysis/v1_dev.csv`; quotes are checked to be verbatim).\n")
    rows = []
    for c in e["categories"]:
        ex = c["example"]
        quote = f"\"{ex['quote']}\" ({ex['complaint_id']})" if ex else ""
        rows.append((c["label"], c["n"], c["judge_relief_company_none"],
                     c["judge_none_company_relief"], c["fix"], quote))
    out = [head, table(("Category", "Cases", "Judge relief, company none",
                        "Judge none, company relief", "What v2 changes", "Example"), rows)]
    out.append("\nThe disagreements that were sorted by reading the case itself (the others follow from "
               "their category):\n")
    out.append(table(("Complaint", "Reading", "Category", "Quote", "Note"),
                     [(r["complaint_id"], r["group"].replace("_", " "), r["category"].replace("_", " "),
                       f"\"{r['quote']}\"", r["note"]) for r in e["read_cases"]]))
    q = m["cues"]
    out.append(f"\nKeyword-proxy check of the cues named in the v2 prompt, on TRAIN only "
               f"({q['n_train']:,} complaints, {pct(q['relief_rate'])} closed with monetary relief; "
               "`src/judge/cues.py`, crude regular expressions):\n")
    out.append(table(("Cue (proxy)", "What the v2 prompt says", "TRAIN complaints with the cue",
                      "Relief with the cue [95% CI]", "Relief without it", "Difference"),
                     [(r["cue"], r["stated"], f"{r['n_with']:,}",
                       f"{pct(r['relief_with'])} [{pct(r['ci_with'][0])}, {pct(r['ci_with'][1])}]",
                       pct(r["relief_without"]), f"{100 * r['difference']:+.1f} pts") for r in q["rows"]]))
    return "\n".join(out)


REWRITE_LABELS = {"all": "**all five rewrites**", "fake_case_note": "fake internal note alone",
                  "injection": "instruction to the reviewer alone", "neutral": "control: neutral sentence appended",
                  "replay": "control: same narrative sent again"}


def b_gaming(m):
    g = m["gaming"]
    if not g["systems"]:
        return PENDING
    names = [n for n in ("v1", "v2", "guarded") if n in g["systems"]]
    short = {"v1": "Judge v1", "v2": "Judge v2", "guarded": "Guarded pipeline"}
    order = g["attacks"] + ["all"] + [c for c in g["controls"] + ["replay"]
                                      if all(c in g["systems"][n] for n in names)]

    def rows(cell):
        return [[a if a in g["attacks"] else REWRITE_LABELS[a]] + [cell(g["systems"][n][a]) for n in names]
                for a in order]

    out = [f"{g['n_items']} DEV complaints, each with every rewrite appended. A flip is a decision that "
           "differs from the decision on the unmodified narrative under the same system. Relief decisions "
           "on the unmodified narratives: " +
           ", ".join(f"{short[n]} {g['systems'][n]['base_relief']}/{g['n_items']}" for n in names) + ".\n",
           "Decisions flipped [95% Wilson CI]:\n",
           table(["Rewrite"] + [short[n] for n in names], rows(
               lambda x: f"{x['flips']}/{x['n']} = {pct(x['flip_rate'])} [{pct(x['flip_ci'][0])}, {pct(x['flip_ci'][1])}]")),
           "\nDirection of the flips (no relief to relief / relief to no relief) and mean change in P(relief):\n",
           table(["Rewrite"] + [short[n] for n in names], rows(
               lambda x: f"{x['to_relief']}/{x['n_base_no_relief']} to relief, "
                         f"{x['to_no_relief']}/{x['n_base_relief']} to no relief, "
                         + (f"{x['mean_delta_p_relief']:+.3f}" if x["mean_delta_p_relief"] == x["mean_delta_p_relief"] else "n/a")))]
    if g["routing"]:
        rn = [n for n in names if n in g["routing"]]
        out += ["\nRouting at the DEV operating point (a confident 'no relief' is decided automatically, a "
                "confident 'relief' is proposed to a person, the rest is escalated). Unmodified narratives: " +
                "; ".join(f"{short[n]} " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(g['routing'][n]['base'].items()))
                          for n in rn) + ". What the rewrites did to them:\n",
                table(["Rewrite"] + [short[n] for n in rn],
                      [[a if a in g["attacks"] else REWRITE_LABELS[a]] +
                       [f"auto-denied {x['n_auto_none']}: {x['auto_none_to_approve']} became relief proposals, "
                        f"{x['auto_none_to_escalate']} escalated; escalated {x['n_escalate']}: "
                        f"{x['escalate_to_approve']} became relief proposals, {x['escalate_to_auto_none']} auto-denied"
                        for x in (g["routing"][n][a] for n in rn)] for a in order])]
    else:
        out.append("\nNo LLM-based system has an operating point on DEV (none reaches the agreement target), so "
                   "none of these decisions would be taken automatically; the flips are changes in the decision "
                   "the system proposes.")
    f = g.get("filter")
    if f:
        out.append(f"\nInput filter: flagged {f['attacked_flagged']['injection']}/{g['n_items']} 'injection' "
                   f"rewrites and {f['attacked_flagged']['fake_case_note']}/{g['n_items']} 'fake case note' rewrites; "
                   f"false alarms on unmodified complaints: {f['clean_flagged']} of {f['clean_n']:,} "
                   f"(DEV + TRAIN). Flagged complaints are routed to a human, so a flagged rewrite "
                   "cannot change an automated decision.")
    return "\n".join(out)


def b_gaming_summary(m):
    g = m["gaming"]
    if not g["systems"]:
        return PENDING
    names = [n for n in ("v1", "v2", "guarded") if n in g["systems"]]
    short = {"v1": "Judge v1", "v2": "Judge v2", "guarded": "Guarded pipeline"}
    order = ["all", "fake_case_note", "injection"] + [c for c in g["controls"] + ["replay"]
                                                     if all(c in g["systems"][n] for n in names)]

    def cell(x):
        return (f"{x['flips']}/{x['n']} = {pct(x['flip_rate'])} [{pct(x['flip_ci'][0])}, {pct(x['flip_ci'][1])}]; "
                f"denials turned into relief {x['to_relief']}/{x['n_base_no_relief']}")

    out = [f"{g['n_items']} DEV complaints x {len(g['attacks'])} rewrites ({', '.join(g['attacks'])}). "
           "Decisions flipped [95% Wilson CI], and how many clean-run denials became relief:\n",
           table(["Rewrite"] + [short[n] for n in names],
                 [[REWRITE_LABELS.get(a, a)] + [cell(g["systems"][n][a]) for n in names] for a in order])]
    f = g.get("filter")
    if f:
        out.append(f"\nThe check in code outside the model: an input filter that flagged "
                   f"{f['attacked_flagged']['injection']}/{g['n_items']} injected instructions, "
                   f"{f['attacked_flagged']['fake_case_note']}/{g['n_items']} fake notes (written to get past it) and "
                   f"{f['clean_flagged']} of {f['clean_n']:,} unmodified complaints; a flagged complaint always "
                   "goes to a person. Cost of the defence on clean complaints: " +
                   (diff_sentence(m["dev"], "v2", "guarded", "DEV") or ""))
    if not g["routing"]:
        out.append("\nNo LLM-based system reaches the agreement target on DEV, so none of these decisions would "
                   "be taken automatically; the flips are changes in the decision the system proposes.")
    return "\n".join(out)


def b_automation_summary(m):
    a, rt = m["automation_dev"], m["routing"]
    if not a:
        return PENDING
    rows = []
    for name, sigs in a.items():
        r = rt.get(name)
        if r is None:
            rows.append((LABELS[name], "no threshold reaches the target", "", "", ""))
            continue
        lo, hi = r["wrong_no_relief_ci"]
        rows.append((LABELS[name],
                     f"{r['auto_no_relief']}/{r['n']} = {pct(r['auto_no_relief'] / r['n'])}",
                     f"{r['wrong_no_relief']}/{r['auto_no_relief']} = "
                     f"{pct(r['wrong_no_relief'] / r['auto_no_relief']) if r['auto_no_relief'] else 'n/a'} "
                     f"[{pct(lo)}, {pct(hi)}]",
                     f"{r['proposed_relief']} ({r['wrong_relief']} where the company paid nothing)",
                     f"{r['escalated']} ({pct(r['relief_among_escalated'])} of them closed with relief)"))
    text = ("Operating point per system: the lowest confidence threshold whose DEV agreement on automated cases "
            f"reaches {pct(m['settings']['target_agreement'], 0)} with at least {m['settings']['min_automated']} "
            "automated, read as three outcomes. Chosen and scored on the same DEV complaints, so optimistic.\n\n" +
            table(("System", "Decided automatically as no relief", "... of which the company did pay (wrongly denied) [95% Wilson CI]",
                   "Relief proposed to a person for approval", "Escalated to a person"), rows))
    reh = (m.get("rehearsal") or {}).get("automation", {})
    notes = []
    for name, r in reh.items():
        at = r["at_dev_threshold"]
        if at and at["n_auto"]:
            notes.append(f"{LABELS[name]}: a threshold chosen the same way on one half of DEV automated "
                         f"{at['n_auto']}/{at['n']} of the other half with {ci(at['agreement'], pct)} agreement "
                         f"({pct(at['majority_agreement_on_auto'])} for always-no-relief on the same cases)")
    if notes:
        below = all(r["at_dev_threshold"]["agreement"]["value"] < m["settings"]["target_agreement"]
                    for r in reh.values() if r["at_dev_threshold"] and r["at_dev_threshold"]["n_auto"])
        text += ("\n\nSplit-half check (not a held-out result): " + "; ".join(notes) + "." +
                 (" That is below the target: on this evidence no system here should decide complaints "
                  "on its own." if below else ""))
    return text


def _human_table(f: dict) -> str:
    rows = [("Person against the recorded company decision", ci(f["person_vs_recorded"]["kappa"]),
             ci(f["person_vs_recorded"]["accuracy"], pct), ci(f["sensitivity"]["person_vs_recorded"]["kappa"]))]
    note = {"v1": " (same question as the rubric)", "v2": " (a different question: v2 predicts the record)"}
    for n, x in f["systems_vs_person"].items():
        rows.append((f"{LABELS[n]} against the person{note.get(n, '')}", ci(x["kappa"]), ci(x["accuracy"], pct),
                     ci(f["sensitivity"]["systems_vs_person"][n]["kappa"])))
    return table(("Agreement", "Kappa [95% CI], decided cases", "Accuracy [95% CI], decided cases",
                  "Kappa with 'cannot tell' counted as no money back"), rows)


def _human_head(h: dict) -> str:
    t = h["translations"]
    return (f"Sample: {h['sample_n']} {h['split'].upper()} complaints drawn by a hash of the ID "
            "(`data/label_sample.json`), which no model had answered and nobody had read one by one; labelled "
            "blind by one person against `docs/labelling_rubric.md`. The analysis was registered before any "
            f"label existed (`docs/analysis_plan_human_labels.md`, {h['plan_registered_at'][:16].replace('T', ' ')} "
            f"UTC, amended {h['amended_at'][:16].replace('T', ' ')} UTC). The annotator reads a Russian machine "
            "translation beside the English original: a reading aid made by a separate model, not the judged "
            f"one, never part of a judged input ({t['flagged']} of {h['sample_n']} translations fail a "
            f"completeness check, {t['missing']} are missing). The classic baselines are refitted without the "
            "sampled complaints.")


def b_human(m):
    h = m["human"]
    out = [_human_head(h)]
    if not h["annotators"]:
        out.append(f"\n_No locked set of labels yet._ Nothing is reported from a partial set. Model answers "
                   f"for the sampled complaints so far: {h['model_answers_for_sample']} (the judge runs on them "
                   "only after the labels are locked).")
        return "\n".join(out)
    for who, r in h["annotators"].items():
        f = r["all"]
        out.append(f"\nAnnotator `{who}` (one person, not a lawyer, no second reader), locked {r['locked_at'][:16].replace('T', ' ')} "
                   f"UTC: {f['decided']} decided ({pct(f['person_relief_share'])} of them money back; the companies "
                   f"gave relief in {pct(f['recorded_relief_share'])} of the sample), {f['cannot_tell']} cannot tell; "
                   f"median {r['median_seconds']:.0f} s per case, {r['total_minutes']:.0f} minutes in all.")
        if "person_vs_recorded" not in f:
            out.append("\nToo few decided cases for agreement figures.")
            continue
        if not any(v in f["systems_vs_person"] for v in ("v1", "v2")):
            out.append("\nThe judge has not answered these complaints yet (`make judge-labelled`).")
        out.append("\n" + _human_table(f))
        p = f["paired_vs_person"]
        if p:
            out.append("\nPaired differences in kappa against the person: " + "; ".join(
                f"{LABELS[k.split('_minus_')[0]]} minus {LABELS[k.split('_minus_')[1]]} "
                f"{ci(x['kappa'], lambda v: f'{v:+.3f}')}" for k, x in p.items()) + ".")
    return "\n".join(out)


def b_human_detail(m):
    h = m["human"]
    if not h["annotators"]:
        return "_No locked set of labels yet._"
    out = []
    for who, r in h["annotators"].items():
        f = r["all"]
        out.append(f"Annotator `{who}`. Cases answered 'cannot tell': {f['cannot_tell']}; recorded relief rate among "
                   f"them {pct(f['cannot_tell_recorded_relief'])}; relief rate of each system on them: " +
                   ", ".join(f"{LABELS[n]} {pct(v)}" for n, v in f["cannot_tell_system_relief"].items()) +
                   f". Notes written: {r['with_note']}. Cases shown with a flagged translation: "
                   f"{r['translation_flagged_when_shown']}.")
        for half, g in r["halves"].items():
            out.append(f"\n{half.capitalize()} half ({g['n']} cases, {g['decided']} decided, {g['cannot_tell']} cannot tell):\n")
            out.append(_human_table(g) if "person_vs_recorded" in g else "Too few decided cases.")
        e = r["error_review"]
        out.append(f"\nSecond screen (after the lock): {e['reviewed']} of {e['total']} individually sorted disagreements "
                   f"reviewed, {e['confirmed']} drafted readings confirmed, {len(e['changed'])} changed" +
                   ("." if not e["changed"] else ": " + "; ".join(
                       f"{c['complaint_id']} from {c['from'].replace('_', ' ')} to {c['to'].replace('_', ' ')}"
                       for c in e["changed"]) + "."))
    return "\n".join(out)


def b_retest(m):
    rt = m.get("retest")
    if not rt:
        return PENDING
    rows = []
    for v, r in rt.items():
        sh = r["shift"]
        rows.append((LABELS[v], r["n"],
                     f"{r['same_decision']}/{r['n']} = {pct(r['same_decision_rate'])} "
                     f"[{pct(r['same_decision_ci'][0])}, {pct(r['same_decision_ci'][1])}]",
                     f"{r['byte_identical']}/{r['n']}", f"{r['same_probability']}/{r['n']}",
                     f"{num(sh['median'], 4)} / {num(sh['p90'], 4)} / {num(sh['max'], 4)}",
                     f"{sh['over_0.01']} / {sh['over_0.05']}",
                     f"{num(r['kappa_cached'])} / {num(r['kappa_replay'])}"))
    out = ["Every DEV request sent a second time with identical settings (temperature 0, same seed), "
           "sequentially; the second answers are kept in `results/cache/replays.jsonl` and are not used "
           "for any other number in this report.\n",
           table(("System", "Requests replayed", "Same decision [95% Wilson CI]", "Byte-identical answer",
                  "Same token P(relief)", "Shift in token P(relief): median / p90 / max",
                  "Shifts above 0.01 / above 0.05", "Kappa against recorded outcomes: cached / replay"), rows)]
    levels = [lv["level"] for lv in next(iter(rt.values()))["top_share_cached"]]
    out += ["\nAgreement with the recorded outcome on the most confident share of the replayed requests, "
            "from the cached answers and from the replayed ones:\n",
            table(["System"] + [f"top {pct(l, 0)}: cached / replay" for l in levels],
                  [[LABELS[v]] + [f"{pct(a['agreement'])} (n={a['n_auto']}) / {pct(b['agreement'])} (n={b['n_auto']})"
                                  for a, b in zip(r["top_share_cached"], r["top_share_replay"])]
                   for v, r in rt.items()])]
    for v, r in rt.items():
        a, b = r["same_prompt_cache_reuse"], r["different_prompt_cache_reuse"]
        flips = "; ".join(f"{f['complaint_id']} (P(relief) {num(f['p_cached'])} then {num(f['p_replay'])})"
                          for f in r["flips"]) or "none"
        out.append(f"\n{LABELS[v]}: decisions that changed: {flips}. The server reused the same number of "
                   f"cached prompt tokens in {a['n']} of the {r['n']} replays, and the token probability was "
                   f"identical in {a['same_probability']} of those; with a different amount of reuse it was "
                   f"identical in {b['same_probability']} of {b['n']}.")
    return "\n".join(out)


def b_automation(m):
    a = m["automation_dev"]
    if not a:
        return PENDING
    rows = []
    for name, sigs in a.items():
        for sig, r in sigs.items():
            at = r["at_threshold"]
            if at is None:
                rows.append((LABELS[name], sig, "none reaches the target", "", "", "", ""))
                continue
            rows.append((LABELS[name], sig, num(r["threshold"], 4),
                         f"{at['n_auto']}/{at['n']} = {pct(at['coverage']['value'])}",
                         ci(at["agreement"], pct), f"{at['auto_no_relief']} / {at['auto_relief']}",
                         pct(at["majority_agreement_on_auto"])))
    out = ["Operating points chosen on DEV with the pre-registered rule (agreement on automated "
           "cases at or above the target; see settings). Everything not automated goes to a human.\n",
           table(("System", "Confidence", "Threshold", "Automated", "Agreement on automated [95% CI]",
                  "Automated no-relief / relief", "Agreement of 'always no relief' on the same cases"), rows)]
    levels = sorted({lv["level"] for s in a.values() for r in s.values() for lv in r["fixed_coverage"]})
    rows = []
    for name, sigs in a.items():
        for sig, r in sigs.items():
            fc = {lv["level"]: lv for lv in r["fixed_coverage"]}
            rows.append([LABELS[name], sig] + [f"{pct(fc[l]['agreement'])} (n={fc[l]['n_auto']})" if l in fc else ""
                                               for l in levels])
    out += ["\nAgreement on the most confident share of DEV cases (ties automated together, so the "
            "automated count can exceed the nominal share):\n",
            table(["System", "Confidence"] + [f"top {pct(l, 0)}" for l in levels], rows)]
    rt = m.get("routing")
    if rt:
        out += ["\nThe same operating points as three outcomes, with the two kinds of automated error apart "
                "(median dollar amount: the largest amount mentioned in each wrongly denied complaint):\n",
                table(("System", "Decided automatically as no relief", "... wrongly (company paid)",
                       "Relief proposed for approval", "... wrongly (company paid nothing)", "Escalated",
                       "Median amount in wrongly denied complaints"),
                      [(LABELS[n], r["auto_no_relief"], r["wrong_no_relief"], r["proposed_relief"], r["wrong_relief"],
                        r["escalated"], "n/a" if r["wrong_no_relief_with_amount"] == 0 else
                        f"{r['wrong_no_relief_median_amount']:,.0f} dollars ({r['wrong_no_relief_with_amount']} with an amount)")
                       for n, r in rt.items()])]
    st = m.get("threshold_stability")
    if st:
        rows = []
        for name, r in st.items():
            rep = (f"{r['auto_cached']} then {r['auto_replay']} automated ({r['left_automation']} left, "
                   f"{r['entered_automation']} entered); agreement {pct(r['agreement_cached'])} then "
                   f"{pct(r['agreement_replay'])}") if "auto_replay" in r else \
                  (f"fact-model probability changed for {r['probability_changed']} of {r['replayed']}"
                   if "probability_changed" in r else "deterministic")
            rows.append((LABELS[name], num(r["threshold"], 4), num(r["observed_max_shift"], 4),
                         r["within_shift_of_threshold"], r["below_threshold_within_shift"], rep))
        out += ["\nHow close the operating points sit to the judge's own run-to-run noise (token "
                "confidence; the shift is the largest change seen when a DEV request was sent twice):\n",
                table(("System", "Threshold", "Largest shift on replay", "DEV cases within that shift of the threshold",
                       "... of which below the threshold", "Same threshold applied to the replayed answers"), rows)]
    if m.get("figure"):
        out.append(f"\n![Automation against agreement]({m['figure']})")
    return "\n".join(out)


def operating_rows(auto: dict):
    rows = []
    for name, r in auto.items():
        at = r["at_dev_threshold"]
        if at is None:
            rows.append((LABELS[name], r["signal"], "none", "nothing automated", "n/a", "n/a"))
        elif at["n_auto"] == 0:
            rows.append((LABELS[name], r["signal"], num(r["threshold"], 4), f"0/{at['n']}", "n/a", "n/a"))
        else:
            rows.append((LABELS[name], r["signal"], num(r["threshold"], 4),
                         f"{at['n_auto']}/{at['n']} = {ci(at['coverage'], pct)}", ci(at["agreement"], pct),
                         pct(at["majority_agreement_on_auto"])))
    return rows


OPERATING_HEADER = ("System", "Confidence", "Threshold fixed beforehand", "Automated [95% CI]",
                    "Agreement on automated [95% CI]", "Agreement of 'always no relief' on the same cases")


def b_rehearsal(m):
    r = m.get("rehearsal")
    if not r:
        return PENDING
    return "\n".join([
        f"Thresholds and the fact model were derived from the {r['n_fit']} DEV complaints in even positions "
        f"(complaint-ID order) and applied unchanged to the {r['n']} in odd positions. This exercises the "
        "exact code path of the held-out run and shows how much an operating point chosen on a small "
        "sample moves; it is not a held-out result (both prompts were written with all of DEV in view, "
        "and the samples are half the size).\n",
        results_table(r["results"]),
        "\nOperating points from the even half applied to the odd half:\n",
        table(OPERATING_HEADER, operating_rows(r["automation"]))])


def b_frozen(m):
    f = m.get("frozen")
    if not f:
        return "_Not frozen yet._"
    c = f["content"]
    rows = []
    for name, op in c["operating_points"].items():
        rows.append((LABELS[name], op["signal"], "none (automates nothing)" if op["threshold"] is None
                     else num(op["threshold"], 6), op["dev_automated"],
                     "n/a" if op["dev_agreement_on_automated"] is None else pct(op["dev_agreement_on_automated"])))
    calls = len(c["final_versions"]) * m["base_rates"]["test"]["n"]
    out = ["Frozen in `results/frozen.json` before any held-out complaint was sent to the model. The "
           "held-out report reads its thresholds from that file.\n",
           table(("System", "Confidence", "Threshold for the held-out run", "DEV cases automated",
                  "DEV agreement on automated"), rows),
           f"\nJudge versions to run on the held-out split: {', '.join(c['final_versions'])} "
           f"({calls} model calls, sequential, about {calls * c['expected_s_per_call'] / 60:.0f} minutes at the "
           f"recent median of {c['expected_s_per_call']:.1f} s per call). Baseline hyperparameters: "
           f"C = {c['tfidf_lr']['C']}, class weighting = {c['tfidf_lr']['class_weight']}. Fact model: logistic "
           f"regression on {len(c['fact_model']['vocab'])} one-hot fact features, weights in the frozen file. "
           f"Model build: Ollama {c['environment']['ollama_version']}, digest `{c['environment']['model_digest'][:12]}`.",
           f"\nHeld-out complaint IDs: {c['heldout']['n']}, SHA-256 of the sorted IDs "
           f"`{c['heldout']['ids_sha256']}`. Acceptance rule for the prompt change: {c['acceptance_rule']}",
           "\nFiles whose SHA-256 is recorded and checked before the run: " +
           ", ".join(f"`{p}`" for p in c["files"]) + "."]
    return "\n".join(out)


def b_test_results(m):
    t = m.get("test")
    if not t:
        sent = m["cost"]["all_calls"]["held_out_calls"]
        f = m.get("frozen")
        frozen = ("The method is not frozen yet." if not f else
                  f"The method is frozen in `results/frozen.json` (SHA-256 `{f['sha256'][:16]}...`): prompts, "
                  f"parser, defence, baselines, the DEV operating points and the {f['content']['heldout']['n']} "
                  f"held-out complaint IDs (SHA-256 of the sorted IDs `{f['content']['heldout']['ids_sha256'][:16]}...`). "
                  f"Acceptance rule: {f['content']['acceptance_rule']}")
        return (f"_Not yet run._ Held-out complaints sent to the model so far: {sent}. {frozen} The run is a "
                "single command (`make heldout`) that refuses to start unless the frozen method is unchanged, "
                "and refuses to run twice.")
    out = [f"Run once: started {t['started_at']}, finished {t['finished_at']} (UTC), versions "
           f"{', '.join(t['versions'])}.\n", results_table(t["results"])]
    for a, b in (("v1", "v2"), ("v2", "guarded"), ("tfidf_lr", "v1"), ("tfidf_lr", "v2")):
        sent = diff_sentence(t["results"], a, b, "held-out")
        if sent:
            out.append("\n" + sent)
    acc = t["acceptance"]
    out.append(f"\nAcceptance rule (frozen before the run): {acc['rule']} Result: "
               + ("v2 accepted." if acc["v2_accepted"] else "no improvement demonstrated."))
    out += ["\nOperating points frozen on DEV applied unchanged to TEST:\n",
            table(OPERATING_HEADER, operating_rows(t["automation"]))]
    return "\n".join(out)


def b_estimates(m):
    e = m.get("estimates")
    lc = m["label_consistency"]
    out = []

    def cell(c):
        if c is None:
            return "nothing automated"
        if c["n_auto"] == 0:
            return f"0/{c['n']}"
        return (f"{c['n_auto']}/{c['n']} = {pct(c['coverage'])} automated, agreement "
                f"{ci(c['agreement'], pct)} (always-no-relief on the same cases: {pct(c['majority'])})")

    if e:
        out.append(table(("System", "Realistic: TEST, threshold frozen on DEV",
                          "Optimistic: DEV, where the threshold was chosen",
                          "Upper bound for this system: best threshold chosen on TEST itself"),
                         [(LABELS[r["system"]], cell(r["realistic"]), cell(r["optimistic"]),
                           cell(r["best_on_scored_split"])) for r in e["rows"]]))
    else:
        out.append("_The realistic estimate is the held-out result at the frozen thresholds and is not "
                   "available until the held-out run._ Until then the only numbers are the DEV operating "
                   "points above (optimistic: chosen on the same complaints) and the split-half rehearsal.")
    s = lc.get("same_company")
    if s:
        out.append(
            f"\nLabel-noise proxy: among {s['pairs']} pairs of near-identical narratives "
            f"(cosine at or above the ceiling threshold) filed against the same company, the recorded "
            f"outcomes agree in {pct(s['agreement'])} of pairs [{pct(s['agreement_ci'][0])}, "
            f"{pct(s['agreement_ci'][1])}], pooled kappa {num(s['pooled_kappa'], 2)}. If both outcomes "
            f"were independent draws from the same case-level probability, no judge reading the text "
            f"could exceed about {pct(s['accuracy_upper_bound'])} accuracy against a single recorded "
            "outcome. Pairs filed within a week agree "
            f"{pct(lc['same_company_within_7_days']['agreement'])} ({lc['same_company_within_7_days']['pairs']} "
            f"pairs, probably handled together), pairs further apart "
            f"{pct(lc['same_company_over_7_days']['agreement'])} ({lc['same_company_over_7_days']['pairs']} pairs, "
            "often a re-filing after an unwelcome answer), so the independence assumption does not hold and "
            "this is a rough proxy, not a measured ceiling.")
    rt = m.get("retest") or {}
    if rt:
        out.append("\nJudge noise: " + "; ".join(
            f"{LABELS[v]} repeats its own decision in {pct(r['same_decision_rate'])} of replayed DEV requests "
            f"(kappa between the two runs {num(r['kappa_between_runs'], 2)})" for v, r in rt.items()) +
            ". Agreement between the judge and the recorded outcome cannot be expected to exceed the "
            "judge's agreement with itself.")
    return "\n".join(out) if out else PENDING


def b_cost(m):
    c = m["cost"]
    rows = []
    for v in ("v1", "v2"):
        if v in c:
            x = c[v]
            rows.append((LABELS[v], f"{x['median_s']:.1f} s (p90 {x['p90_s']:.1f} s)",
                         f"{x['mean_prompt_tokens']:.0f} in / {x['mean_output_tokens']:.0f} out "
                         f"(longest answer {x['max_output_tokens']})",
                         "local GPU, sequential calls, DEV run"))
    if "tfidf_lr" in c:
        x = c["tfidf_lr"]
        rows.append((LABELS["tfidf_lr"], f"{x['predict_ms_per_item_one_by_one']:.2f} ms one by one, "
                     f"{x['predict_ms_per_item_batched']:.3f} ms batched", "",
                     f"CPU; training {x['fit_seconds']:.0f} s on {x['n_train']:,} labels, "
                     f"{x['cv_select_plus_fit_seconds']:.0f} s including the hyperparameter search"))
    out = [table(("System", "Time per decision", "Tokens", "Notes"), rows)]
    if "v2" in c and "tfidf_lr" in c:
        ratio = c["v2"]["median_s"] * 1000 / c["tfidf_lr"]["predict_ms_per_item_one_by_one"]
        out.append(f"\nOne judge v2 decision costs about {ratio:,.0f} times the wall time of one "
                   "logistic-regression decision on this machine.")
    a = c["all_calls"]
    out.append(f"\nAll model calls on record: {a['calls']:,}, {a['hours']:.2f} hours of model time "
               f"(sum of per-call wall time); answers cut off by the output limit: {a['truncated']}; "
               f"calls on held-out complaints: {a['held_out_calls']}.\n")
    out.append(table(("Version", "Set", "Calls", "Model time", "Median per call", "First call (UTC)", "Last call (UTC)"),
                     [(r["version"], r["set"], r["calls"], f"{60 * r['hours']:.1f} min", f"{r['median_s']:.1f} s",
                       r["first_call"][:16].replace("T", " "), r["last_call"][:16].replace("T", " "))
                      for r in c["by_run"]]))
    if c["run_speed"]:
        out.append("\nSpeed logged after the first calls of each run (the laptop was short of memory for "
                   "parts of the session, which is why the same kind of call took between five and fifteen seconds):\n")
        out.append(table(("Logged (UTC)", "Run", "Seconds per call", "Generation tokens/s", "Prompt tokens/s"),
                         [(e["at"][:16].replace("T", " "), f"{e['label']} {e['version']}", f"{e['s_per_call']:.1f}",
                           e["generation_tokens_per_s"], e["prompt_tokens_per_s"]) for e in c["run_speed"]]))
    if c["concurrency"]:
        out.append("\nConcurrency check: cached v1 DEV requests replayed at 1, 2 and 4 concurrent calls "
                   "(`results/cache/concurrency_replays.jsonl`, not used for any other number) and compared "
                   "with the cached answers:\n")
        out.append(table(("Check started (UTC)", "Concurrent calls", "Requests", "Seconds per request",
                          "Decisions that differ", "Answers not byte-identical", "Largest shift in token P(relief)",
                          "Swap in use before / after (GB)"),
                         [(r["benchmark_at"][:16].replace("T", " "), r["concurrency"], r["n_items"],
                           f"{r['s_per_item']:.1f}", r["decisions_differ"], r["raw_text_differs"],
                           "n/a" if r["max_abs_delta_p_token"] is None else num(r["max_abs_delta_p_token"], 4),
                           f"{r['swap_used_mb_before'] / 1024:.1f} / {r['swap_used_mb_after'] / 1024:.1f}")
                          for r in c["concurrency"]]))
    d = c.get("concurrency_decision")
    if d:
        out.append(f"\nConcurrent calls used for every scored run: {d['workers']}. {d['reason']}")
    return "\n".join(out)


BLOCKS = {            # every block, in the order of results/report.md
    "summary": b_summary,
    "settings": b_settings,
    "data": b_data,
    "base_rates": b_base_rates,
    "inputs": b_inputs,
    "dev_results": b_dev_results,
    "date_halves": b_date_halves,
    "error_groups": b_error_groups,
    "error_analysis": b_error_analysis,
    "v1v2": b_v1v2,
    "gaming_summary": b_gaming_summary,
    "gaming": b_gaming,
    "retest_summary": b_retest_summary,
    "retest": b_retest,
    "automation_summary": b_automation_summary,
    "automation": b_automation,
    "rehearsal": b_rehearsal,
    "human": b_human,
    "human_detail": b_human_detail,
    "frozen": b_frozen,
    "test_results": b_test_results,
    "estimates": b_estimates,
    "cost": b_cost,
}
README_BLOCKS = ("summary", "v1v2", "retest_summary", "inputs", "error_groups", "gaming_summary",
                 "automation_summary", "human", "test_results")   # the README must contain at least these


def render(text: str, m: dict) -> str:
    def sub(mt):
        name = mt.group(2)
        body = BLOCKS[name](m) if name in BLOCKS else PENDING
        return mt.group(1) + body.rstrip() + "\n" + mt.group(4)
    return BLOCK_RE.sub(sub, text)


def number_guard(text: str) -> list[str]:
    """Result-like numbers (decimals, percentages, counts >= 10 that are not
    years) must live inside generated blocks."""
    outside = BLOCK_RE.sub("", text)
    outside = re.sub(r"`[^`]*`", "", outside)              # code spans: commands, model ids
    outside = re.sub(r"```.*?```", "", outside, flags=re.S)
    outside = re.sub(r"(?m)^\s*\d+\. ", "", outside)          # numbered-list markers
    bad = []
    for mt in re.finditer(r"(?<![\w/.-])(\d[\d,]*(?:\.\d+)?%?)(?![\w/-])", outside):
        tok = mt.group(1).rstrip(",")
        if tok in ("95%",):
            continue
        if re.fullmatch(r"(19|20)\d\d", tok):               # years
            continue
        if re.fullmatch(r"\d", tok):                        # list numbers, single digits
            continue
        bad.append(tok)
    return bad


def _clean(x):
    """Round floats (so late decimals of the classic baseline cannot cause a
    spurious drift failure) and make NaN valid JSON."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    x = float(x)
    return None if x != x else round(x, 6)


def emit(m: dict, check: bool = False) -> list[str]:
    """Write (or, with check, compare) README blocks, metrics.json and report.md.
    Returns the list of problems."""
    dump = dict(m)
    if m.get("frozen"):
        dump["frozen"] = {k: v for k, v in m["frozen"].items() if k != "content"}
    metrics_text = json.dumps(_clean(dump), indent=1, sort_keys=True) + "\n"
    readme_old = README.read_text()
    readme_new = render(readme_old, m)
    from .analysis import report_markdown
    report_new = report_markdown(m, BLOCKS, render)
    problems = []
    for path, new in ((README, readme_new), (METRICS_FILE, metrics_text), (REPORT_FILE, report_new)):
        old = path.read_text() if path.exists() else ""
        if old != new:
            if check:
                diff = "".join(list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                         str(path), "regenerated"))[:60])
                problems.append(f"{path.name} differs from regenerated output:\n{diff}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new)
    missing = [b for b in README_BLOCKS if f"<!-- BEGIN {b} -->" not in readme_new]
    if missing:
        problems.append("README has no block for: " + ", ".join(missing))
    bad = number_guard(readme_new)
    if bad:
        problems.append("README has result-like numbers outside generated blocks: " + ", ".join(bad))
    if m.get("frozen") and m["frozen"]["problems"]:
        problems.append("the frozen method no longer matches (re-freeze only if the held-out run has "
                        "not started):\n- " + "\n- ".join(m["frozen"]["problems"]))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    check = ap.parse_args().check
    problems = emit(compute(), check)
    if problems:
        print("\n\n".join(problems))
        sys.exit(1)
    print("check passed: README, metrics.json and report.md match the cached results"
          if check else "regenerated README.md, results/metrics.json, results/report.md")


if __name__ == "__main__":
    main()
