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
def _auc_b() -> int:
    import inspect

    from . import metrics as M
    return inspect.signature(M.bootstrap_stat).parameters["B"].default


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
        ("Bootstrap", f"{s['bootstrap_b']} resamples for kappa, accuracy, paired differences and coverage; "
                      f"{_auc_b()} for AUC intervals (the default of `metrics.bootstrap_stat`, as frozen); seed "
                      f"{s['bootstrap_seed']}, percentile 95% CI"),
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
    "keyword_rule": "Keyword rule (cues from the DEV reading, choice on TRAIN)",
    "company_rule": "Company rule (planned: issuer's TRAIN relief rate above one half)",
    "issuer_rate": "Issuer relief rate (added later: above the TRAIN base rate)",
    "tfidf_masked": "TF-IDF + logistic regression, issuer names masked (added later)",
    "tfidf_lr": "TF-IDF + logistic regression (trained on TRAIN)",
    "v1": "LLM judge v1 (plain rubric)",
    "v2": "LLM judge v2 (predicts the record, facts first)",
    "guarded": "Guarded pipeline (v2 facts -> fact model; filter routes to human)",
}
READS = {"majority": "nothing", "keyword_rule": "narrative", "company_rule": "company field only",
         "issuer_rate": "company field only", "tfidf_masked": "header + narrative, issuer names masked",
         "tfidf_lr": "header + narrative", "v1": "header + narrative", "v2": "header + narrative",
         "guarded": "header + narrative"}


def kappa_cell(r: dict) -> str:
    """Kappa with its interval; a constant prediction has kappa zero by definition."""
    if r["predicted_positive_rate"] in (0.0, 1.0):
        return "0.000 (constant answer)"
    return ci(r["kappa"])


def results_table(res: dict) -> str:
    rows = []
    for name, r in res.items():
        if name not in LABELS:
            continue
        rows.append((LABELS[name], r["n"], ci(r["accuracy"]), kappa_cell(r),
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
            f"The first-named system agrees with the record on {t['only_second_agrees']} complaints where the "
            f"other does not ({t['gained_no_relief_cases']} closed without relief, {t['gained_relief_cases']} with), "
            f"and disagrees on {t['only_first_agrees']} where the other agrees ({t['lost_relief_cases']} closed "
            f"with relief, {t['lost_no_relief_cases']} without).")


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
    for a, b in (("v1", "v2"), ("v2", "guarded"), ("tfidf_lr", "v1"), ("tfidf_lr", "v2"), ("issuer_rate", "v1"),
                 ("issuer_rate", "v2"), ("company_rule", "v1"), ("company_rule", "v2"), ("tfidf_masked", "tfidf_lr"),
                 ("v1", "tfidf_masked"), ("v2", "tfidf_masked")):
        sent = diff_sentence(d, a, b, "DEV")
        if sent:
            out.append("\n" + sent)
    out.append("\nNo acceptance rule for v2 was written before this DEV comparison; the rule for the "
               "held-out comparison is in the frozen file.")
    out.append("\n" + b_issuer(m))
    k = m["keyword_rule"]
    out.append(f"\nKeyword rule: the candidate cues come from the reading of DEV disagreements; of "
               f"{len(k['candidates'])} candidate rules scored on TRAIN, the best was "
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
    if name in ("majority", "company_rule", "issuer_rate"):
        return "a table lookup"
    if name == "keyword_rule" and "tfidf_lr" in c:
        return f"{c['tfidf_lr']['keyword_rule_ms_per_item']:.1f} s (CPU)"
    if name in ("tfidf_lr", "tfidf_masked") and "tfidf_lr" in c:
        return f"{c['tfidf_lr']['predict_ms_per_item_one_by_one']:.1f} s (CPU)"
    v = "v2" if name == "guarded" else name
    if v in c:
        return f"about {c[v]['hours_per_1000']:.0f} h (one laptop GPU)"
    return "n/a"


def b_summary(m):
    b = m["base_rates"]
    t = m.get("test")
    head = (f"DEV: {b['dev']['n']} complaints, {pct(b['dev']['base_rate'])} closed with monetary relief. "
            f"Held-out TEST: {b['test']['n']} complaints, {pct(b['test']['base_rate'])}"
            + ("" if t else ", not yet run") +
            f". TRAIN (classic baselines only): {b['train']['n']:,}, {pct(b['train']['base_rate'])}. "
            "Kappa is Cohen's kappa against the recorded company decision, with a 95% bootstrap interval; "
            "AUC is the chance that a relief case is scored above a no-relief case, over all pairs and over "
            "pairs answered by the same issuer.\n")
    rank = m["ranking_dev"]["systems"]
    rows = []
    for name in SUMMARY_ROWS:
        if name not in m["dev"]:
            continue
        r = m["dev"][name]
        op = (m["automation_dev"].get(name) or {}).get("token")
        if name not in m["automation_dev"]:
            auto = "not assessed"
        elif op["at_threshold"] is None:
            auto = "none: no threshold reaches the target"
        else:
            at = op["at_threshold"]
            lo, hi = at["agreement_wilson"]
            auto = (f"{at['n_auto']}/{at['n']} at {pct(at['agreement']['value'])} agreement (Wilson "
                    f"[{pct(lo)}, {pct(hi)}]); chosen and scored on the same cases, does not survive a split")
        auc = (f"{num(rank[name]['auc']['value'], 2)} / {num(rank[name]['within_issuer_auc']['value'], 2)}"
               if name in rank else "n/a")
        held = kappa_cell(t["results"][name]) if t and name in t["results"] else "not run"
        rows.append((LABELS[name], READS[name], kappa_cell(r), auc, auto, held, time_per_1000(m, name)))
    people = F._people(m)
    tail = ("\n\nThe blind human labels are a TRAIN sample, so the person never appears in a DEV column; the person's "
            "table against the recorded decision prints once, in the human-label section." if people else
            "\n\nThe blind human labels are a TRAIN sample, so the person never appears in a DEV column; see the "
            "human-label sections.")
    return head + "\n" + table(("System", "Reads", "DEV kappa [95% CI]", "DEV AUC: all pairs / same issuer",
                                "Automated on DEV at the planned target", "Held-out kappa [95% CI]",
                                "Time per 1,000 decisions"), rows) + tail


SUMMARY_ROWS = ("majority", "keyword_rule", "company_rule", "issuer_rate", "tfidf_masked", "tfidf_lr", "v1", "v2",
                "guarded")


def _versus(x: dict, a: str, b: str) -> str:
    k = x["kappa"]
    rel = "is below" if k["hi"] < 0 else "is above" if k["lo"] > 0 else "is not distinguishable from"
    return f"{a} {rel} {b} ({ci(k, lambda v: f'{v:+.3f}')})"


def verdicts(m) -> list[str]:
    """Plain-language reading of the DEV table, derived from the intervals."""
    d, rk = m["dev"], m["ranking_dev"]
    ranked = sorted((n for n in d if n in LABELS), key=lambda n: -d[n]["kappa"]["value"])
    out = [f"Highest DEV kappa: {LABELS[ranked[0]]}."]
    parts = []
    for v, vn in (("v1", "Judge v1"), ("v2", "Judge v2")):
        for base, bn in (("tfidf_lr", "TF-IDF"), ("issuer_rate", "the issuer relief rate"),
                         ("company_rule", "the planned company rule")):
            x = d.get(f"{v}_minus_{base}")
            if x:
                parts.append(_versus(x, vn, bn))
    out.append("Paired kappa differences: " + "; ".join(parts) + ".")
    r, iss = rk["systems"], m["issuer"]
    dv1, dv2 = rk["differences"]["tfidf_lr_minus_v1"], rk["differences"]["tfidf_lr_minus_v2"]
    out.append(
        f"What the TF-IDF lead consists of: the issuer's TRAIN relief rate alone, without reading the complaint, "
        f"orders DEV complaints as well as TF-IDF does (AUC {num(r['issuer_rate']['auc']['value'], 2)} against "
        f"{num(r['tfidf_lr']['auc']['value'], 2)}; difference "
        f"{ci(rk['differences']['issuer_rate_minus_tfidf_lr']['auc'], lambda v: f'{v:+.3f}')}), and TF-IDF's "
        f"strongest features are issuer names ({pct(iss['train_narratives_naming_an_issuer'], 0)} of TRAIN "
        f"narratives name an issuer). Among complaints answered by the same issuer "
        f"({m['ranking_dev']['within_issuer_pairs']} pairs) the lead of TF-IDF over the judge is no longer "
        f"demonstrated: AUC difference {ci(dv1['within_issuer_auc'], lambda v: f'{v:+.3f}')} against v1 and "
        f"{ci(dv2['within_issuer_auc'], lambda v: f'{v:+.3f}')} against v2 (all pairs: "
        f"{ci(dv1['auc'], lambda v: f'{v:+.3f}')} and {ci(dv2['auc'], lambda v: f'{v:+.3f}')}). The judge's own "
        f"ordering is weak either way (AUC {num(r['v1']['auc']['value'], 2)} and {num(r['v2']['auc']['value'], 2)}).")
    x = d.get("v2_minus_v1")
    if x:
        k = x["kappa"]
        out.append(("The prompt revision from v1 to v2 raised kappa." if k["lo"] > 0 else
                    "The prompt revision from v1 to v2 lowered kappa." if k["hi"] < 0 else
                    f"The prompt revision from v1 to v2 did not improve kappa ({ci(k, lambda v: f'{v:+.3f}')}).") +
                   f" Accuracy rose only because v2 predicts relief for {pct(d['v2']['predicted_positive_rate'])} of "
                   f"complaints where v1 predicted it for {pct(d['v1']['predicted_positive_rate'])}; "
                   f"recall of relief fell from {num(d['v1']['recall_1']['value'], 2)} to "
                   f"{num(d['v2']['recall_1']['value'], 2)}.")
    sh = m["split_half"].get("tfidf_lr")
    if sh:
        out.append(f"No judge configuration reaches the planned automation target on DEV. The one operating "
                   f"point in the table (TF-IDF) is in-sample: in {sh['splits']:,} random half-splits of DEV a "
                   f"threshold existed in {pct(sh['threshold_found'] / sh['splits'], 0)} of them, and the other "
                   f"half reached the target in {sh['other_half_reaches_target']} (median agreement there "
                   f"{pct(sh['median_agreement_other_half'], 0)}).")
    return out


def b_v1v2(m):
    return "\n\n".join(verdicts(m))


def b_issuer(m):
    i, rk = m["issuer"], m["ranking_dev"]
    rows = [(LABELS[n], ci(x["auc"]), ci(x["within_issuer_auc"]) if n != "issuer_rate" else "0.500 (constant within an issuer)")
            for n, x in rk["systems"].items()]
    return "\n".join([
        f"Issuer identity (added after DEV results). TRAIN has {i['issuers_in_train']} issuers; among those with "
        f"at least 100 TRAIN complaints the smoothed relief rate (the issuer lookup's score) runs from "
        f"{pct(i['lowest'][1])} ({i['lowest'][0]}) to {pct(i['highest'][1])} ({i['highest'][0]}); "
        f"{int(i['issuers_above_half'])} issuer with at least 20 TRAIN complaints paid more than half of them (raw "
        "rate), which is why the planned company rule rarely predicts relief. TF-IDF features with the largest weights towards "
        f"relief: {', '.join(i['top_features']['towards_relief'])}; towards no relief: "
        f"{', '.join(i['top_features']['towards_no_relief'])}. The masked model replaces {i['issuer_words_masked']} "
        "issuer words (chosen by a share-and-lift rule on TRAIN) and a short list of common short names.\n",
        table(("System", "AUC, all pairs [95% CI]", "AUC, pairs with the same issuer [95% CI]"), rows),
        "\nDifferences (first minus second): " + "; ".join(
            f"{k.replace('_minus_', ' minus ').replace('_', ' ')}: all pairs {ci(x['auc'], lambda v: f'{v:+.3f}')}, "
            f"same issuer {ci(x['within_issuer_auc'], lambda v: f'{v:+.3f}')}" for k, x in rk["differences"].items()) + "."])


def _tfidf_lead(m) -> str:
    """What the TF-IDF lead over the judges consists of, in the words the intervals allow."""
    d, rk = m["dev"], m["ranking_dev"]
    out = f"Both versions are below a TF-IDF baseline (kappa {num(d['tfidf_lr']['kappa']['value'], 2)})."
    issuer = rk["differences"].get("issuer_rate_minus_tfidf_lr")
    if issuer and issuer["auc"]["lo"] <= 0 <= issuer["auc"]["hi"]:
        out += (" The issuer's relief rate alone, without reading the complaint, orders complaints about as well "
                f"(AUC {num(rk['systems']['issuer_rate']['auc']['value'], 2)} against "
                f"{num(rk['systems']['tfidf_lr']['auc']['value'], 2)}), so part of that lead may be knowing how "
                "each issuer usually answers")
    masked = [d.get(f"tfidf_masked_minus_{v}") for v in ("v1", "v2")]
    if all(masked):
        diffs = " and ".join(ci(x["kappa"], lambda v: f"{v:+.2f}") for x in masked)
        sep = "; but" if out.endswith("answers") else "."
        if all(x["kappa"]["lo"] > 0 for x in masked):
            out += f"{sep} with issuer names masked, TF-IDF is still above both judges (kappa difference {diffs})"
        else:
            out += f"{sep} with issuer names masked, its lead over the judges is not demonstrated (kappa difference {diffs})"
    return out + ". " if not out.endswith(".") else out + " "


def b_lead(m):
    """The result, stated before anything else."""
    d = m["dev"]
    k = d["v2_minus_v1"]["kappa"]
    below = all(d[f"{v}_minus_tfidf_lr"]["kappa"]["hi"] < 0 for v in ("v1", "v2"))
    none = all(m["automation_dev"][n]["token"]["threshold"] is None for n in ("v1", "v2", "guarded"))
    where = ("the held-out results are below" if m.get("test") else "the held-out split has not been run")
    head = (f"DEV result ({d['v1']['n']} complaints, used for every choice" if m.get("test")
            else f"Result so far (DEV, {d['v1']['n']} complaints")
    return (f"**{head}; {where}).** The judge "
            f"agrees with recorded company decisions little better than chance: kappa {num(d['v1']['kappa']['value'], 2)} "
            f"for the first prompt and {num(d['v2']['kappa']['value'], 2)} for the revised one. The revision "
            + ("did not help" if k["lo"] <= 0 else "helped") +
            f" ({ci(k, lambda v: f'{v:+.2f}')}). " +
            (_tfidf_lead(m) if below else "") +
            ("No judge configuration qualifies for automation at the planned target. " if none else "") +
            ("Blind human labels on a separate TRAIN sample are reported in the human-label section."
             if m["human"]["annotators"] else
             "What a careful person would decide from the same text is not known yet; blind human labels are "
             "being collected."))


def b_deviation_facts(m):
    i, f = m["issuer"], m["data"]["funnel"]
    g = i["first_commit_grid"]
    return (f"Numbers behind the deviations list: the first commit's grid would have chosen C = {g['chosen']['C']} without "
            f"class weighting, DEV kappa {num(g['dev_kappa'])}, against {num(m['dev']['tfidf_lr']['kappa']['value'])} "
            f"for the grid that follows the plan text. Label cross-check against the live API: "
            f"{f['dropped_metadata_mismatch_vs_live_api']} of {f['slice_with_narrative']:,} complaints with a narrative "
            f"differed and were dropped, {f.get('dropped_not_in_live_api', 0)} were missing.")


def b_inputs(m):
    st = m["settings"]
    return (f"Narratives are cut at {st['max_narrative_chars']:,} characters for every system. "
            f"Model: `{st['model']}`, temperature {st['options']['temperature']}, seed {st['options']['seed']}.")


def _who_read(m) -> str:
    e = m["error_analysis"]
    pv = e["provenance"]
    t = lambda k: pv[k][11:16]                      # noqa: E731  (UTC hour and minute)
    rev = [r["error_review"] for r in m["human"]["annotators"].values() if r["error_review"]["reviewed"]]
    return (f"The categories, quotes and notes were drafted by {pv['produced_by']}; " +
            (f"when they were drafted the author had read {pv['cases_read_by_the_author']} of the cases"
             if rev else f"the author has read {pv['cases_read_by_the_author']} of the cases") +
            f". The {e['n_disagreements']} narratives hold "
            f"{e['words_in_disagreements']:,} words. The v1 run ended at {t('v1_dev_run_finished')} UTC, the "
            f"disagreements were exported at {t('disagreements_exported_for_reading')}, the v2 prompt was first "
            f"used at {t('v2_prompt_first_used')} and the category file was last written at "
            f"{t('category_file_last_written')}: the v2 prompt rests on a machine pass of minutes over these "
            "cases, not on a person's reading. " +
            (_second_screen_sentence(m, rev[0]) +
             f" The drafted readings of the other {e['n_disagreements'] - rev[0]['reviewed']} of the "
             f"{e['n_disagreements']} disagreements are still the assistant's and unconfirmed." if rev else
             "The author's own reading of the cases sorted one by one will be reported separately when it exists."))


def _second_screen_sentence(m, rev) -> str:
    """What the record of the second screen shows (frontpage._went_through: count, timing, notes)."""
    who = next(w for w, r in m["human"]["annotators"].items() if r["error_review"] is rev)
    n, t, notes = F._went_through(rev, F._screen(m, who))
    return (f"After the blind labels were locked, the author went through {n} cases that the assistant had sorted one "
            f"by one, on a second screen{t}: he confirmed {rev['confirmed']} drafted readings, changed {len(rev['changed'])}{notes} (case by "
            "case in the human detail section).")


def _denial(m) -> str:
    e = m["error_analysis"]
    d, c = e["prior_denial"], e["category_only"]
    return (f"The other {e['n_category_only']} disagreements carry a category only ("
            + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) +
            "). These categories do not show that a case cannot be decided from its text: judge v2 extracts "
            f"'the issuer had already said no' for {d['n_denied']} of {e['n_dev']} DEV complaints (relief rate "
            f"{pct(d['relief_rate_denied'])} with it, {pct(d['relief_rate_other'])} without), and judge v1 agreed "
            f"with the record on {d['v1_agrees_among_denied']} of those {d['n_denied']} "
            f"({d['v1_agrees_among_denied_relief']} of them closed with relief). That the record often turns on "
            "evidence outside the complaint is a hypothesis here, which the blind human labels " +
            (_bear_on(m) if m["human"]["annotators"] else "are meant to bear on") +
            "; it is not a finding.")


def _bear_on(m) -> str:
    f = next(iter(m["human"]["annotators"].values()))["all"]
    return (f"bear on (the annotator could not decide {f['cannot_tell']} of {f['n']}; see the human-label section), "
            "though no test of the hypothesis was registered")


def b_error_groups(m):
    e = m.get("error_analysis")
    if not e:
        return PENDING
    rows = []
    for g in e["groups"]:
        ex = g["example"]
        rows.append((g["label"], g["n"],
                     (f"\"{ex['quote']}\" ({ex['complaint_id']}" + (f": {ex['note']})" if ex["note"] else ")"))
                     if ex else ""))
    return (f"Judge v1 disagreed with the record on {e['n_disagreements']} of {e['n_dev']} DEV complaints. "
            f"{_who_read(m)}\n\n{e['n_read']} disagreements were sorted one by one into a three-way reading:\n\n" +
            table(("Reading as drafted (the author's review of each case is in the human detail section)"
                   if any(r["error_review"]["reviewed"] for r in m["human"]["annotators"].values())
                   else "Reading (drafted, not yet confirmed by the author)", "Cases", "Example"), rows) +
            "\n\n" + _denial(m))


def b_retest_summary(m):
    rt = m.get("retest")
    if not rt:
        return PENDING
    return " ".join(
        f"Asked twice with identical settings ({r['n']} of {r['n_split']} DEV requests replayed), {LABELS[v]} "
        f"gave the same decision in {r['same_decision']}/{r['n']} = "
        f"{pct(r['same_decision_rate'])} [{pct(r['same_decision_ci'][0])}, {pct(r['same_decision_ci'][1])}] "
        f"(largest change in P(relief) {num(r['shift']['max'], 2)}; kappa against the record "
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
            "(`results/error_analysis/v1_dev.csv`; quotes are checked to be verbatim). " + _who_read(m) + "\n")
    rows = []
    for c in e["categories"]:
        ex = c["example"]
        quote = f"\"{ex['quote']}\" ({ex['complaint_id']})" if ex else ""
        rows.append((c["label"], c["n"], c["judge_relief_company_none"],
                     c["judge_none_company_relief"], c["fix"], quote))
    out = [head, table(("Category", "Cases", "Judge relief, company none",
                        "Judge none, company relief", "What v2 changes", "Example"), rows)]
    out.append("\n" + _denial(m))
    out.append("\nThe disagreements that were sorted one by one into the three-way reading:\n")
    out.append(table(("Complaint", "Reading", "Category", "Quote", "Note"),
                     [(r["complaint_id"], r["group"].replace("_", " "), r["category"].replace("_", " "),
                       f"\"{r['quote']}\"", r["note"]) for r in e["read_cases"]]))
    q = m["cues"]
    out.append("\nThe v2 prompt names seven cues. Its issuer-mistake cue has no TRAIN proxy (it comes from "
               "four DEV cases), and the interval of the service-quality proxy contains the rate without the cue, "
               "so 'checked on TRAIN' holds for the other five only.")
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
SHORT = {"v1": "Judge v1", "v2": "Judge v2", "guarded": "Guarded pipeline"}


def _gaming_notes(m) -> list[str]:
    g, f = m["gaming"], m["gaming"].get("filter")
    names = [n for n in ("v1", "v2", "guarded") if n in g["systems"]]
    out = ["Each rewrite is a single fixed sentence appended to the narrative, so every row is one template "
           "applied to " + str(g["n_items"]) + " complaints, not a family of attacks. Rows for one rewrite use a "
           "Wilson interval; the pooled row repeats every complaint five times, so its interval resamples "
           "complaints."]
    above = {n: [a for a in g["attacks"] if g["systems"][n][a].get("vs_neutral", {}).get("p", 1) < 0.05]
             for n in names}
    out.append("Against the neutral-sentence control on the same complaints (exact McNemar test, p below 0.05), "
               "the rewrites that flip more decisions are: " +
               "; ".join(f"{SHORT[n]}: {', '.join(above[n]) or 'none'}" for n in names) + ". 'None' is a statement "
               "about power, not robustness: " + "; ".join(
                   f"{SHORT[n]} starts from {g['n_items'] - g['systems'][n]['base_relief']} denials and the fake note "
                   f"turned {g['systems'][n]['fake_case_note']['to_relief']} of them into relief"
                   for n in names if not above[n]) + ". Ten tests are uncorrected here; the exploratory section gives "
               "the Holm correction.")
    d = m["dev"]
    if "guarded" in names:
        out.append(f"The guarded pipeline answers relief for {d['guarded']['confusion']['relief_relief'] + d['guarded']['confusion']['none_relief']} "
                   f"of {d['guarded']['n']} DEV complaints and for {g['systems']['guarded']['base_relief']} of these "
                   f"{g['n_items']}; a policy that always answers no relief has no flips at all, so its low flip "
                   "rate says little. Its fact model is fitted on DEV, including these complaints.")
    if f:
        reused = [a for a, k in f["answer_reused"].items() if k == g["n_items"]]
        out.append(f"The check in code outside the model is an input filter: it flagged {f['attacked_flagged']['injection']}/"
                   f"{g['n_items']} copies of the one injected sentence, {f['attacked_flagged']['fake_case_note']}/"
                   f"{g['n_items']} fake notes, {f['clean_flagged']} of {f['clean_n']:,} unmodified complaints, and "
                   f"{f['paraphrases_flagged']} of {f['paraphrases']} other wordings of the same instruction "
                   "(written afterwards with the patterns in view; `src/judge/gaming.py`). A flagged complaint goes "
                   "to a person. " +
                   (f"The guarded pipeline's zero for '{', '.join(reused)}' holds by construction: the filter "
                    "removes the appended sentence and the cached answer to the clean narrative is reused, so the "
                    "interval describes nothing. " if reused else "") +
                   "A defence tuned to known attacks overstates robustness.")
    if not g["routing"]:
        out.append("No LLM-based system reaches the agreement target on DEV, so none of these decisions would "
                   "be taken automatically; the flips are changes in the decision the system proposes.")
    return out


def _flip_cell(x, by_construction=False):
    if by_construction:
        return f"{x['flips']}/{x['n']} (by construction)"
    return f"{x['flips']}/{x['n']} = {pct(x['flip_rate'])} [{pct(x['flip_ci'][0])}, {pct(x['flip_ci'][1])}]"


def _denial_cell(x):
    return (f"{x['to_relief']}/{x['n_base_no_relief']}" +
            (f" = {pct(x['to_relief_rate'], 0)}" if x["n_base_no_relief"] else ""))


def b_gaming(m):
    g = m["gaming"]
    if not g["systems"]:
        return PENDING
    names = [n for n in ("v1", "v2", "guarded") if n in g["systems"]]
    order = g["attacks"] + ["all"] + [c for c in g["controls"] + ["replay"]
                                      if all(c in g["systems"][n] for n in names)]
    reused = {a for a, k in (g.get("filter") or {}).get("answer_reused", {}).items() if k == g["n_items"]}

    def rows(cell):
        return [[a if a in g["attacks"] else REWRITE_LABELS[a]] +
                [cell(g["systems"][n][a], n == "guarded" and a in reused) for n in names] for a in order]

    out = [f"{g['n_items']} DEV complaints, each with every rewrite appended. A flip is a decision that "
           "differs from the decision on the unmodified narrative under the same system. Relief decisions "
           "on the unmodified narratives: " +
           ", ".join(f"{SHORT[n]} {g['systems'][n]['base_relief']}/{g['n_items']}" for n in names) + ".\n",
           "Decisions flipped [95% CI]:\n",
           table(["Rewrite"] + [SHORT[n] for n in names], rows(_flip_cell)),
           "\nDenials on the unmodified narrative that became relief (the two judges start from very "
           "different numbers of denials, so the pooled flip rates above are not comparable):\n",
           table(["Rewrite"] + [SHORT[n] for n in names], rows(lambda x, _: _denial_cell(x))),
           "\nRelief turned into no relief, and mean change in P(relief):\n",
           table(["Rewrite"] + [SHORT[n] for n in names], rows(
               lambda x, _: f"{x['to_no_relief']}/{x['n_base_relief']}, " +
               (f"{x['mean_delta_p_relief']:+.3f}" if x["mean_delta_p_relief"] == x["mean_delta_p_relief"] else "n/a"))),
           "\nComparison with the neutral control (complaints flipped only by the rewrite / only by the control, "
           "exact McNemar p):\n",
           table(["Rewrite"] + [SHORT[n] for n in names],
                 [[a] + [f"{x['only_rewrite']} / {x['only_control']}, p = {x['p']:.3f}"
                         for x in (g["systems"][n][a]["vs_neutral"] for n in names)] for a in g["attacks"]])]
    if g["routing"]:
        rn = [n for n in names if n in g["routing"]]
        out += ["\nRouting at the DEV operating point (a confident 'no relief' is decided automatically, a "
                "confident 'relief' is proposed to a person, the rest is escalated). Unmodified narratives: " +
                "; ".join(f"{SHORT[n]} " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(g['routing'][n]['base'].items()))
                          for n in rn) + ". What the rewrites did to them:\n",
                table(["Rewrite"] + [SHORT[n] for n in rn],
                      [[a if a in g["attacks"] else REWRITE_LABELS[a]] +
                       [f"auto-denied {x['n_auto_none']}: {x['auto_none_to_approve']} became relief proposals, "
                        f"{x['auto_none_to_escalate']} escalated; escalated {x['n_escalate']}: "
                        f"{x['escalate_to_approve']} became relief proposals, {x['escalate_to_auto_none']} auto-denied"
                        for x in (g["routing"][n][a] for n in rn)] for a in order])]
    return "\n".join(out + ["\n" + "\n\n".join(_gaming_notes(m))])


def b_gaming_summary(m):
    g = m["gaming"]
    if not g["systems"]:
        return PENDING
    names = [n for n in ("v1", "v2", "guarded") if n in g["systems"]]
    order = ["all", "fake_case_note", "injection"] + [c for c in g["controls"] + ["replay"]
                                                     if all(c in g["systems"][n] for n in names)]
    reused = {a for a, k in (g.get("filter") or {}).get("answer_reused", {}).items() if k == g["n_items"]}

    def cell(x, by_construction):
        return _flip_cell(x, by_construction) + f"; denials turned into relief {_denial_cell(x)}"

    out = [f"{g['n_items']} DEV complaints x {len(g['attacks'])} rewrites ({', '.join(g['attacks'])}). "
           "Decisions flipped [95% CI], and how many denials on the unmodified narrative became relief "
           "(compare the judges on the denials: they start from " +
           " and ".join(f"{g['n_items'] - g['systems'][n]['base_relief']}" for n in names[:2]) + " denials):\n",
           table(["Rewrite"] + [SHORT[n] for n in names],
                 [[REWRITE_LABELS.get(a, a)] + [cell(g["systems"][n][a], n == "guarded" and a in reused)
                                                for n in names] for a in order])]
    return "\n".join(out + ["\n" + "\n\n".join(_gaming_notes(m))])


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
            table(("System", "Decided automatically as no relief", "... of which the company did pay [95% Wilson CI]",
                   "Relief proposed to a person for approval", "Escalated to a person"), rows))
    notes = []
    for name, sh in m["split_half"].items():
        if sh["threshold_found"]:
            notes.append(f"{LABELS[name]}: a threshold in {sh['threshold_found']} of {sh['splits']:,} random "
                         f"half-splits, the other half at the target in {sh['other_half_reaches_target']} "
                         f"(median agreement {pct(sh['median_agreement_other_half'], 0)})")
        else:
            notes.append(f"{LABELS[name]}: no threshold in any of {sh['splits']:,} random half-splits")
    if notes:
        text += ("\n\nThe same rule applied to one random half of DEV and scored on the other: " +
                 "; ".join(notes) + "." +
                 (" On this evidence no system here should decide complaints on its own."
                  if not any(sh["other_half_reaches_target"] for sh in m["split_half"].values()) else ""))
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
            "(`data/label_sample.json`), which the judged model had not answered and the annotator had not "
            "seen; labelled blind by one person against `docs/labelling_rubric.md`. The analysis was registered "
            "before any "
            f"label existed (`docs/analysis_plan_human_labels.md`, {h['plan_registered_at'][:16].replace('T', ' ')} "
            f"UTC, amended {h['amended_at'][:16].replace('T', ' ')} UTC). The annotator reads a Russian machine "
            "translation beside the English original: a reading aid made by a separate model, not the judged "
            "one (Anthropic's `claude-fable-5-2`, run as sub-agents of the AI coding assistant working on this "
            f"repository), never part of a judged input ({t['flagged']} of {h['sample_n']} translations fail a "
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
        out.append(f"\nSecond screen (after the lock): {e['reviewed']} of the {e['total']} disagreements that the assistant "
                   f"had sorted one by one reviewed, {e['confirmed']} drafted readings confirmed, {len(e['changed'])} changed" +
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
    out = ["DEV requests sent a second time with identical settings (temperature 0, same seed), sequentially: " +
           "; ".join(f"{LABELS[v]} {r['n']} of {r['n_split']}" for v, r in rt.items()) +
           " (for v1 the gaming subset first, then DEV order). The second answers are kept in "
           "`results/cache/replays.jsonl` and are not used for any other number in this report. Kappa against "
           "the record on the replayed subset shows how unstable kappa is at this size: on its replayed "
           "subset v1 scores far from its figure on all of DEV.\n",
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
           table(("System", "Confidence", "Threshold", "Automated", "Agreement on automated [95% percentile-bootstrap CI]",
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
                "(median dollar amount: the largest amount mentioned in each auto-denied complaint the company paid):\n",
                table(("System", "Decided automatically as no relief", "... where the company paid",
                       "Relief proposed for approval", "... where the company paid nothing", "Escalated",
                       "Median amount in the auto-denied complaints the company paid"),
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
        out.append(f"\n![Automation against agreement]({m['figure'].rsplit('/', 1)[-1]})")   # report.md is in results/
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
                         f"{at['n_auto']}/{at['n']} = {ci(at['coverage'], pct)}",
                         f"{pct(at['agreement']['value'])} [{pct(at['agreement_wilson'][0])}, "
                         f"{pct(at['agreement_wilson'][1])}]", pct(at["majority_agreement_on_auto"])))
    return rows


OPERATING_HEADER = ("System", "Confidence", "Threshold fixed beforehand", "Automated [95% CI]",
                    "Agreement on automated [95% Wilson CI]", "Agreement of 'always no relief' on the same cases")


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
        table(OPERATING_HEADER, operating_rows(r["automation"])),
        "\nThe held-out decision rules applied to this rehearsal (to show what the held-out section will "
        "print; one even/odd split is arbitrary, see the repeated half-splits above):\n",
        "\n".join(f"- {c['verdict']}" + (f" ({ci(c['interval'], lambda v: f'{v:+.3f}')})" if c["interval"] else "")
                  for c in r["claims"])])


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
    out = [f"Frozen in `results/frozen.json` (SHA-256 `{f['sha256']}`) before any held-out complaint was sent "
           "to the model. The held-out report reads its thresholds and decision rules from that file. A local "
           "file proves nothing about when it was written: " +
           ("the commit was not published before the run, but its hash and the file's SHA-256 were written to a "
            "GitHub gist before the run (see Freeze record)." if m.get("test") else
            "the commit that contains it is to be published before the run.") + "\n",
           table(("System", "Confidence", "Threshold for the held-out run", "DEV cases automated",
                  "DEV agreement on automated"), rows),
           f"\nJudge versions to run on the held-out split: {', '.join(c['final_versions'])} "
           f"({calls} model calls for the scored pass and {calls} for the noise pass, sequential, about "
           f"{2 * calls * c['expected_s_per_call'] / 60:.0f} minutes at the recent median of "
           f"{c['expected_s_per_call']:.1f} s per call). Baseline hyperparameters: C = {c['tfidf_lr']['C']}, class "
           f"weighting = {c['tfidf_lr']['class_weight']}; keyword rule: \"{c['keyword_rule']['rule']}\". Fact model: "
           f"logistic regression on {len(c['fact_model']['vocab'])} one-hot fact features, weights in the frozen "
           f"file. Model build: Ollama {c['environment']['ollama_version']}, digest "
           f"`{c['environment']['model_digest'][:12]}`. Held-out complaint IDs: {c['heldout']['n']}, SHA-256 of the "
           f"sorted IDs `{c['heldout']['ids_sha256']}`. Answer caches at the freeze: " +
           "; ".join(f"`{k}` {v['bytes']:,} bytes, SHA-256 `{v['sha256'][:16]}...`" for k, v in c["caches"].items()) +
           " (append-only: later answers may be added, these bytes may not change).",
           "\nWhat will be said about the held-out split, and the rule that decides it:\n",
           "\n".join(f"- `{x['id']}`: {x['rule']}" for x in c["claims"]),
           f"\nFiles whose SHA-256 is recorded and checked before the run ({len(c['files'])}): " +
           ", ".join(f"`{p}`" for p in c["files"]) + "."]
    for x in f.get("post_run_changes") or []:
        was = f"`{x['frozen_sha256'][:16]}...`" if x["frozen_sha256"] else "none, added after the freeze"
        out.append(f"\nChanged after the held-out run and accepted only because `results/post_run_changes.json` "
                   f"records it: `{x['file']}` (SHA-256 at the freeze {was}, now "
                   f"`{x['sha256'][:16]}...`): {x['what']}")
    return "\n".join(out)


def _requests_sentence(t) -> str:
    q = t.get("requests")
    if not q:
        return ""
    again = q["sent_more_than_once"]
    return (f"Requests recorded before sending (`results/cache/sent.jsonl`): {q['scored']} in the scored pass and "
            f"{q['noise']}, each under its own label, in the noise pass; " +
            ("none was sent twice." if not again else
             f"sent again after a first send that returned no answer: {', '.join(again)}."))


def b_test_results(m):
    t = m.get("test")
    if not t:
        sent = m["cost"]["all_calls"]["held_out_calls"]
        f = m.get("frozen")
        frozen = ("The method is not frozen at this commit." if not f else
                  f"The method is frozen in `results/frozen.json` (SHA-256 `{f['sha256'][:16]}...`): every source "
                  f"file, the data, the plans, the DEV operating points, the {f['content']['heldout']['n']} held-out "
                  f"complaint IDs (SHA-256 of the sorted IDs `{f['content']['heldout']['ids_sha256'][:16]}...`) and "
                  f"{len(f['content']['claims'])} claims with their decision rules (listed in the full report).")
        return (f"_Not yet run._ Held-out complaints sent to the model so far: {sent}. {frozen} The run is a "
                "single command (`make heldout`) that refuses to start unless the frozen method is unchanged "
                "and committed, and refuses to run twice; a second pass over the same requests is stored apart "
                "and reported as noise only.")
    out = [f"Run once from commit `{t['commit'][:12]}`: started {t['started_at']}, finished {t['finished_at']} "
           f"(UTC), versions {', '.join(t['versions'])}. " + _requests_sentence(t) + "\n", results_table(t["results"]),
           "\nThe frozen claims, each decided by its rule:\n",
           "\n".join(f"- {c['verdict']}" + (f" ({ci(c['interval'], lambda v: f'{v:+.3f}')})" if c["interval"] else "")
                     for c in t["claims"])]
    rk = t["ranking"]
    out += ["\nOrdering of held-out complaints:\n",
            table(("System", "AUC, all pairs [95% CI]", "AUC, pairs with the same issuer [95% CI]"),
                  [(LABELS[n], ci(x["auc"]), ci(x["within_issuer_auc"]) if n != "issuer_rate"
                    else "0.500 (constant within an issuer)") for n, x in rk["systems"].items()])]
    for a, b in (("v1", "v2"), ("tfidf_lr", "v1"), ("tfidf_lr", "v2"), ("issuer_rate", "v1"), ("issuer_rate", "v2")):
        sent = diff_sentence(t["results"], a, b, "held-out")
        if sent:
            out.append("\n" + sent)
    out += ["\nOperating points frozen on DEV applied unchanged to TEST:\n",
            table(OPERATING_HEADER, operating_rows(t["automation"]))]
    if t["noise"]:
        out.append("\nNoise pass (every held-out request sent a second time; never scored against the record): " +
                   "; ".join(f"{LABELS[v]} repeated its decision in {r['same_decision']}/{r['n']} = "
                             f"{pct(r['same_decision_rate'])} [{pct(r['same_decision_ci'][0])}, "
                             f"{pct(r['same_decision_ci'][1])}], kappa between the two passes "
                             f"{num(r['kappa_between_runs'], 2)}, largest shift in P(relief) {num(r['shift']['max'], 2)}, "
                             f"byte-identical answers {r['byte_identical']}/{r['n']}"
                             for v, r in t["noise"].items()) + ". The DEV retest above and this noise pass differ in how "
                   "often the answers were byte-identical; memory was not recorded during either, so the cause is not "
                   "known.")
    else:
        out.append("\nThe noise pass has not finished yet.")
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
        from .metrics import wilson
        k = round(c["agreement"]["value"] * c["n_auto"])
        lo, hi = wilson(k, c["n_auto"])
        return (f"{c['n_auto']}/{c['n']} = {pct(c['coverage'])} automated, agreement "
                f"{ci(c['agreement'], pct)} (percentile bootstrap; Wilson [{pct(lo)}, {pct(hi)}]) "
                f"(always-no-relief on the same cases: {pct(c['majority'])})")

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
        ratio = c["v2"]["mean_s"] * 1000 / c["tfidf_lr"]["predict_ms_per_item_one_by_one"]
        out.append(f"\nA judge decision took between {c['v2']['min_s_all_calls']:.0f} and "
                   f"{c['v2']['max_s_all_calls']:.0f} seconds in one evening on this laptop, so the ratio to one "
                   f"logistic-regression decision is only an order of magnitude: about {round(ratio, -3):,.0f} "
                   f"times. Per 1,000 decisions (DEV mean): {c['v1']['hours_per_1000']:.1f} h for v1 and "
                   f"{c['v2']['hours_per_1000']:.1f} h for v2.")
    a = c["all_calls"]
    out.append(f"\nAll model calls on record: {a['calls']:,}, {a['hours']:.2f} hours of model time "
               f"(sum of per-call wall time); answers cut off by the output limit: {a['truncated']}; "
               f"calls on held-out complaints: {a['held_out_calls']}." +
               (f" Not counted here: {c['translation_calls_not_counted']} calls in which the judged model translated "
                "an earlier labelling sample into Russian (run log, event 'translation'); those translations "
                "were discarded without being shown to the annotator." if c.get("translation_calls_not_counted")
                else "") + "\n")
    out.append(table(("Version", "Set", "Calls", "Model time", "Median per call", "First call (UTC)", "Last call (UTC)"),
                     [(r["version"], r["set"], r["calls"], f"{60 * r['hours']:.1f} min", f"{r['median_s']:.1f} s",
                       r["first_call"][:16].replace("T", " "), r["last_call"][:16].replace("T", " "))
                      for r in c["by_run"]]))
    if c["run_speed"]:
        out.append("\nSpeed logged after the first calls of each run (the same kind of call took between five and "
                   "fifteen seconds; swap in use was recorded only during the concurrency check below, so the cause of "
                   "the spread is not known):\n")
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


# ------------------------------------------------- first screen (added after the held-out run)
# The post-run presentation lives in frontpage.py and the exploratory analyses in posthoc.py; both
# were added after the held-out run and are recorded in results/post_run_changes.json.
from . import frontpage as F  # noqa: E402  (imported here: it reads the helpers above at call time)

b_headline, b_practice = F.b_headline, F.b_practice


# ------------------------------------------- changes after the held-out run (presentation only)
POST_RUN_FILE = C.RESULTS / "post_run_changes.json"
# the only files a post-run change may touch: the presentation code, the exploratory module, the
# deviations note (the last two did not exist at the freeze: their frozen hash is recorded as null), and the
# Makefile, whose venv target only gained a check of the Python version (third change). Sixth change: the SQL
# loader and its walk-through (both new, frozen hash null; they feed no report number), and the Makefile's `sql`
# target
PRESENTATION_FILES = ("src/judge/report.py", "src/judge/frontpage.py", "src/judge/posthoc.py", "docs/deviations.md",
                      "Makefile", "src/judge/sql.py", "docs/sql.md")


def accept_post_run_changes(m: dict) -> dict:
    """After the held-out run the method can no longer be re-frozen. A later change to a presentation
    file is accepted only when results/post_run_changes.json records it with the file's hash at the
    freeze (as in results/frozen.json; null for a file added after the freeze) and its hash now;
    every other difference from the frozen method stays a problem and fails the check."""
    f = m.get("frozen")
    if not f or not m.get("test") or not POST_RUN_FILE.exists():
        return m
    from .freeze import sha256_path
    records = {r["file"]: r for r in json.loads(POST_RUN_FILE.read_text())}
    suffix = " changed since the freeze"
    accepted, rest = [], []
    for p in f["problems"]:
        name = p[: -len(suffix)] if p.endswith(suffix) else None
        r = records.get(name)
        if (r is not None and name in PRESENTATION_FILES and (C.ROOT / name).exists()
                and r["frozen_sha256"] == f["content"]["files"].get(name)
                and r["sha256"] == sha256_path(C.ROOT / name)):
            accepted.append({k: r[k] for k in ("file", "frozen_sha256", "sha256", "what")})
        else:
            rest.append(p)
    return {**m, "frozen": {**f, "problems": rest, "post_run_changes": accepted}}


BLOCKS = {            # every block, in the order of results/report.md
    "first_screen": F.b_first_screen,
    "who": F.b_who,
    "headline": b_headline,
    "practice": b_practice,
    "exploratory": F.b_exploratory,
    "lead": b_lead,
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
    "issuer": b_issuer,
    "gaming_summary": b_gaming_summary,
    "gaming": b_gaming,
    "retest_summary": b_retest_summary,
    "retest": b_retest,
    "automation_summary": b_automation_summary,
    "automation": b_automation,
    "rehearsal": b_rehearsal,
    "human": F.b_human,
    "human_detail": F.b_human_detail,
    "frozen": b_frozen,
    "freeze_record": F.b_freeze_record,
    "test_results": b_test_results,
    "estimates": b_estimates,
    "cost": b_cost,
    "deviation_facts": b_deviation_facts,
    "data_licence": F.b_data_licence,
    "sql_queries": F.b_sql_queries,     # sixth change, after the held-out run: a pointer to sql/
    "deviations_in_short": F.b_deviations_in_short,   # sixth change: the count of post-run changes from the records
}
README_BLOCKS = ("first_screen", "who", "headline", "practice", "inputs", "human", "freeze_record",
                 "deviation_facts", "data_licence", "sql_queries",
                 "deviations_in_short")   # the README must contain at least these


def render(text: str, m: dict, root: str = "") -> str:
    """Fill every block of text. root is the path from the rendered file to the repository root, for the relative
    links the blocks write (seventh change): empty for the README, "../" for results/report.md."""
    mb = {**m, F.LINK_BASE: root} if root else m

    def sub(mt):
        name = mt.group(2)
        body = BLOCKS[name](mb) if name in BLOCKS else PENDING
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
    m = accept_post_run_changes(m)
    dump = dict(m)
    if m.get("frozen"):
        dump["frozen"] = {k: v for k, v in m["frozen"].items() if k != "content"}
    metrics_text = json.dumps(_clean(dump), indent=1, sort_keys=True) + "\n"
    readme_old = README.read_text()
    readme_new = render(readme_old, m)
    from .analysis import report_markdown
    report_new = report_markdown(m, BLOCKS, lambda text, mm: render(text, mm, root="../"))   # one level down
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
    integrity = (m.get("posthoc") or {}).get("integrity") or {}
    if integrity.get("problems"):
        problems.append("the record of the held-out run no longer matches:\n- " + "\n- ".join(integrity["problems"]))
    if m.get("frozen") and m["frozen"]["problems"]:
        problems.append("the frozen method no longer matches (re-freeze only if the held-out run has "
                        "not started):\n- " + "\n- ".join(m["frozen"]["problems"]))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    check = ap.parse_args().check
    from . import posthoc

    m = compute()
    m["posthoc"] = posthoc.compute(m)             # exploratory, after the held-out run
    problems = emit(m, check)
    if problems:
        print("\n\n".join(problems))
        sys.exit(1)
    print("check passed: README, metrics.json and report.md match the cached results"
          if check else "regenerated README.md, results/metrics.json, results/report.md")


if __name__ == "__main__":
    main()
