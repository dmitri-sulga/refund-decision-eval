"""The README's first screen, the held-out claims, the practitioner section, the exploratory section,
the freeze record and the human-label section. Written after the held-out run; presentation only
(results/post_run_changes.json records it). Numbers come from the frozen computation (analysis.py)
or, where marked exploratory, from posthoc.py.
"""
from __future__ import annotations



class _Report:
    """The helpers of report.py, looked up when called (report.py imports this module)."""

    def __getattr__(self, name):
        from . import report
        return getattr(report, name)


R = _Report()

SHORT = {
    "majority": "Always no relief",
    "keyword_rule": "Keyword rule (added later)",
    "company_rule": "Company rule (planned)",
    "issuer_rate": "Issuer relief rate (added later)",
    "tfidf_masked": "TF-IDF, issuer names masked (added later)",
    "tfidf_lr": "TF-IDF + logistic regression",
    "v1": "LLM judge v1 (plain rubric)",
    "v2": "LLM judge v2 (predicts the record)",
    "guarded": "Guarded pipeline (v2 facts -> fact model)",
}
READS = {"majority": "nothing", "keyword_rule": "narrative", "company_rule": "issuer only",
         "issuer_rate": "issuer only", "tfidf_masked": "text, issuer names masked", "tfidf_lr": "text",
         "v1": "text", "v2": "text", "guarded": "text"}
NAME = {"majority": "always answering no relief", "keyword_rule": "keyword rule", "company_rule": "planned company rule",
        "issuer_rate": "issuer lookup", "tfidf_masked": "masked TF-IDF", "tfidf_lr": "TF-IDF model", "v1": "judge v1",
        "v2": "judge v2", "guarded": "guarded pipeline"}
TEXT_SYSTEMS = ("tfidf_lr", "tfidf_masked", "v1", "v2", "guarded")
EXPL = "*(exploratory)*"
ATTACK_WORDS = {"promise": "claimed promise", "emotion": "emotional pressure", "legal_threat": "legal threat",
                "injection": "instruction to the reviewer", "fake_case_note": "fake internal note",
                "neutral": "neutral sentence (control)", "names_towards_relief": "issuer names weighted towards relief",
                "names_towards_no_relief": "issuer names weighted towards no relief (control)"}
GAPS = ("fabricated merchant correspondence", "fake prior approvals with reference numbers",
        "claims written to match the published rubric, cues and fact-model weights",
        "a merchant dispute reframed as fraud through the sub-issue the consumer selects (every system reads it)",
        "other languages and obfuscated text sent to the model", "the attack at the start, in the middle or inside a "
        "quoted email, and padding that pushes real content past the narrative cut",
        "attacks aimed at the fact-extraction step, named as such", "using the filter's flag to choose routing",
        "refiling with trivial edits until the answer changes", "an attack set written by someone other than the "
        "defence author and held out from tuning")


def _claim_status(c: dict) -> str:
    """met / not met / undetermined / decided, read off the verdict that the frozen rule produced."""
    v = c["verdict"]
    if c["id"] == "v2_vs_v1":
        return "met" if v == "v2 replaces v1" else "not met"
    if c["id"] == "tfidf_operating_point":
        return ("met" if "meets the target" in v and "not shown" not in v else
                "not met" if "misses the target" in v else "undetermined")
    if c["id"] in ("judge_vs_baseline", "within_issuer"):
        return "undetermined" if ("not distinguishable" in v or "not demonstrated" in v) else "decided"
    return "met"


def _automated_cell(t: dict, name: str) -> str:
    a = t["automation"].get(name)
    if a is None:
        return "not assessed"
    at = a["at_dev_threshold"]
    if at is None:
        return "none (no DEV threshold)"
    if at["n_auto"] == 0:
        return f"0/{at['n']}"
    return f"{at['n_auto']}/{at['n']}, agreement {R.pct(at['agreement']['value'])} ({_wilson(at)})"


def _sg(x: dict, d: int = 3) -> str:
    return R.ci(x, lambda v: f"{v:+.{d}f}")


def _pts(x: dict) -> str:
    return f"{100 * x['value']:+.1f} points [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]"


def _wilson(at: dict) -> str:
    return f"Wilson [{R.pct(at['agreement_wilson'][0])}, {R.pct(at['agreement_wilson'][1])}]"


def _pays(r: dict) -> int:
    return r["confusion"]["relief_relief"] + r["confusion"]["none_relief"]


def _errors(r: dict) -> str:
    return f"{r['confusion']['none_relief']} / {r['confusion']['relief_none']}"


def _ready(m) -> bool:
    return bool(m.get("test") and m.get("posthoc"))


def _issuer_share(m) -> tuple[str, float]:
    si = m["posthoc"]["sub_issues"]
    key = next(k for k in si if "isn't resolving" in k)
    return key, si[key]["n"] / sum(v["n"] for v in si.values())


# ------------------------------------------------------------------ first screen
def b_first_screen(m):
    if not _ready(m):
        return R.PENDING
    t, ph = m["test"], m["posthoc"]
    r, n = t["results"], t["n"]
    acc = ph["heldout"]["accuracy_minus_always_no"]
    best = max(acc, key=lambda k: r[k]["accuracy"]["value"])
    _, share = _issuer_share(m)
    pdn = m["error_analysis"]["prior_denial"]
    text_best = max(("tfidf_lr", "v1", "v2", "guarded"), key=lambda k: r[k]["kappa"]["value"])
    sy, df = t["ranking"]["systems"], t["ranking"]["differences"]
    chance = all(sy[k]["within_issuer_auc"]["lo"] <= 0.5 for k in TEXT_SYSTEMS if k in sy)
    v2n = m["gaming"]["systems"]["v2"]["fake_case_note"]
    at = t["automation"]["tfidf_lr"]["at_dev_threshold"]
    toward_pay = [k for k in ("v2", "guarded") if r[k]["confusion"]["none_relief"] < r[k]["confusion"]["relief_none"]]
    out = [
        f"**The data.** Complaints to the US regulator (the CFPB) about disputed credit-card charges, mostly filed "
        f"after the card issuer had handled the dispute itself: {R.pct(share, 0)} of the "
        f"{m['base_rates']['train']['n']:,} training complaints are filed as 'company isn't resolving a dispute', and "
        f"judge v2 reads a prior refusal in {pdn['n_denied']} of {m['error_analysis']['n_dev']} development complaints. "
        "The label is how the company closed the complaint with the regulator, with money back (goodwill credits "
        "included) or without, so every figure here is agreement with what the company did, not correctness.",
        "",
        f"- **Always answering \"no money back\" matches the record in {R.pct(r['majority']['accuracy']['value'])} of "
        f"the {n} held-out complaints; the best system, the {NAME[best]}, matches it in "
        f"{R.pct(r[best]['accuracy']['value'])}, a difference of {_pts(acc[best])}.**",
        f"- **A model applying the written refund rules to the consumer's account (judge v1) would pay "
        f"{_pays(r['v1'])} of the {n}; the companies paid {r['majority']['confusion']['relief_none']}.** It errs "
        f"towards paying ({r['v1']['confusion']['none_relief']} payments the company did not make, "
        f"{r['v1']['confusion']['relief_none']} refusals where it paid)" +
        (f"; {' and '.join(('the ' if k == 'guarded' else '') + NAME[k] for k in toward_pay)} err towards refusing "
         f"({'; '.join(_errors(r[k]) for k in toward_pay)})." if toward_pay else "."),
        f"- **The record follows which issuer answered** {EXPL}. A lookup of each issuer's past payout rate, which "
        f"never reads the complaint, agrees with the record about as well as the best text model (kappa "
        f"{R.num(r['issuer_rate']['kappa']['value'])} against {R.num(r[text_best]['kappa']['value'])}, difference "
        f"{_sg(ph['heldout']['tfidf_minus_issuer_kappa'])}); hiding issuer names costs the text model ranking quality "
        f"(AUC {_sg(df['tfidf_masked_minus_tfidf_lr']['auc'])})" +
        ("; and among complaints about the same issuer no system that reads the text ranks paid above unpaid "
         "clearly better than a coin toss." if chance else "."),
        f"- **Decision: automate nothing.** No system is shown to reach the planned "
        f"{R.pct(m['settings']['target_agreement'], 0)} agreement on the cases it would decide alone "
        f"(TF-IDF: {at['n_auto']}/{at['n']} at {R.pct(at['agreement']['value'])}, {_wilson(at)}), and in a test on "
        f"development complaints a fake internal note pasted into the text turned {v2n['to_relief']} of "
        f"{v2n['n_base_no_relief']} of judge v2's refusals into refunds.",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ frozen claims and the table
def _claims(m) -> str:
    t, st, ph = m["test"], m["settings"], m["posthoc"]
    cl = t["claims"]
    out = []
    for c in cl:
        if c["id"] == "v2_vs_v1":
            out.append(f"- *v2 replaces v1* (only if the paired kappa difference lies entirely above zero): "
                       f"**{_claim_status(c)}**, {_sg(c['interval'])}.")
    jb = [c for c in cl if c["id"] == "judge_vs_baseline"]
    dec = [c for c in jb if _claim_status(c) == "decided"]
    und = [c for c in jb if _claim_status(c) != "decided"]
    bon = ph["heldout"]["bonferroni_judge_vs_baseline"]
    key = {"the TF-IDF model": "tfidf_lr", "the issuer relief rate": "issuer_rate", "the company rule": "company_rule"}
    held = [k for k in bon if bon[k]["hi"] < 0]
    out.append(f"- *Each judge against TF-IDF, the issuer relief rate and the company rule* ({len(jb)} paired kappa "
               "differences): **decided:** " + "; ".join(f"{c['verdict']} ({_sg(c['interval'])})" for c in dec) +
               (f"; both also exclude zero at the {R.pct(bon[held[0]]['level'], 2)} level that corrects for "
                f"{len(bon)} comparisons {EXPL}" if len(held) == len(dec) and dec else "") +
               ". **Undetermined**, the interval includes zero: " + "; ".join(
                   f"{c['verdict'].replace(' is not distinguishable from ', ' against ')} ({_sg(c['interval'])})"
                   for c in und) + ".")
    wi = [c for c in cl if c["id"] == "within_issuer"]
    out.append(f"- *TF-IDF orders complaints better than the judge within one issuer*: " + "; ".join(
        f"**{_claim_status(c)}** against {c['verdict'].split(' over ')[1].split(' is ')[0]} "
        f"(AUC difference {_sg(c['interval'])})" if "over" in c["verdict"] else
        f"**{_claim_status(c)}**, {c['verdict']} ({_sg(c['interval'])})" for c in wi) + ".")
    at = t["automation"]["tfidf_lr"]["at_dev_threshold"]
    for c in cl:
        if c["id"] == "tfidf_operating_point":
            out.append(f"- *The TF-IDF threshold frozen on DEV meets the {R.pct(st['target_agreement'], 0)} agreement "
                       f"target on held-out cases*: **{_claim_status(c)}**, {c['verdict'].split(': ', 1)[-1]} "
                       f"({at['n_auto']}/{at['n']} automated, {R.pct(at['agreement']['value'])}, {_wilson(at)}; "
                       f"always answering no relief agrees on {R.pct(at['majority_agreement_on_auto'])} of the same "
                       "cases).")
    nz = t["noise"]
    out.append("- Two frozen items cannot fail and are listed for completeness: the LLM configurations automate "
               "nothing, because none had a DEV threshold; and the noise pass was done (" + ", ".join(
                   f"judge {v} repeated {r['same_decision']}/{r['n']} decisions" for v, r in nz.items()) + ").")
    return "\n".join(out)


def _table(m) -> str:
    t, d, b = m["test"], m["dev"], m["base_rates"]
    rows = []
    for name in R.SUMMARY_ROWS:
        if name not in d or name not in t["results"]:
            continue
        r = t["results"][name]
        rows.append((SHORT[name], READS[name], R.pct(r["accuracy"]["value"]), f"{_pays(r)} ({_errors(r)})",
                     R.kappa_cell(r), R.kappa_cell(d[name]), _automated_cell(t, name), R.time_per_1000(m, name)))
    paid = t["results"]["majority"]["confusion"]["relief_none"]
    return (R.table(("System", "Reads", f"Matches the record, held-out (n={b['test']['n']})",
                     f"Would pay, of {b['test']['n']} (where the company did not / refusals where it paid); "
                     f"the companies paid {paid}",
                     "Held-out kappa [95% CI]", f"DEV kappa [95% CI], n={b['dev']['n']}",
                     "Automated on held-out at the frozen threshold", "Time per 1,000 decisions (DEV timings)"), rows) +
            f"\n\nKappa is agreement beyond what the base rate gives (0 for a constant answer). Relief base rate "
            f"{R.pct(b['test']['base_rate'])} held-out, {R.pct(b['dev']['base_rate'])} DEV. Text: the product and "
            "issue fields and the consumer's narrative; issuer only: the issuer's relief rate on the "
            f"{b['train']['n']:,} TRAIN complaints. DEV was used for every choice after the first prompt, so the DEV "
            "column is optimistic.")


def _people(m):
    return [(who, r) for who, r in m["human"]["annotators"].items() if "person_vs_recorded" in r["all"]]


def person_table(m) -> str:
    """The person against the recorded decision next to every system, on the same decided labelled cases.
    Never mixed into a DEV or held-out column: the labelled sample is drawn from TRAIN."""
    out = []
    for who, r in _people(m):
        f = r["all"]
        same = f["systems_vs_recorded_same_cases"]
        rows = [("Person (blind, one annotator)", R.ci(f["person_vs_recorded"]["kappa"]),
                 R.pct(f["person_vs_recorded"]["accuracy"]["value"]), R.pct(f["person_relief_share"]))]
        rows += [(SHORT[n], R.ci(x["kappa"]), R.pct(x["accuracy"]["value"]), R.pct(x["predicted_positive_rate"]))
                 for n, x in same.items() if n in SHORT]
        out.append(f"Against the recorded decision on the same {f['decided']} decided cases of the {f['n']}-complaint "
                   f"TRAIN labelling sample (`{who}`; {f['cannot_tell']} answered 'cannot tell' = "
                   f"{R.pct(f['cannot_tell'] / f['n'], 0)}, left out here; baselines refitted without these complaints):\n\n"
                   + R.table(("Who", "Kappa against the record [95% CI]", "Matches the record", "Pays"), rows))
    return "\n\n".join(out)


def b_headline(m):
    t = m.get("test")
    if not t:
        return R.b_lead(m)
    if not m.get("posthoc"):
        return R.PENDING
    out = [f"**Held-out run**: {t['n']} complaints, scored once from commit `{t['commit'][:7]}`. That commit froze the "
           "method; it precedes the run in this repository's history, and a GitHub gist created before the run holds "
           "its hashes (see the freeze section below). **Claims frozen before the run**, each with the outcome its "
           "rule in `results/frozen.json` dictates (the outcome words were chosen after the run):\n", _claims(m),
           "\n" + _table(m)]
    out.append("\n" + person_table(m) if _people(m) else
               "\nBlind human labels are being collected; nothing is reported until the set is complete and locked.")
    return "\n".join(out)


# ------------------------------------------------------------------ practitioner reading
def b_practice(m):
    if not _ready(m):
        return R.PENDING
    t, ph, g, c, st = m["test"], m["posthoc"], m["gaming"], m["cost"], m["settings"]
    r, h = t["results"], ph["heldout"]
    sy, df = t["ranking"]["systems"], t["ranking"]["differences"]
    wi = {k: sy[k]["within_issuer_auc"] for k in TEXT_SYSTEMS if k in sy}
    top = max(wi, key=lambda k: wi[k]["lo"])
    w10 = h["v2_within_issuer_at_10000"]
    dv = m["ranking_dev"]["systems"]["tfidf_lr"]["within_issuer_auc"]
    at = t["automation"]["tfidf_lr"]["at_dev_threshold"]
    sl = h["tfidf_slice"]
    op = m["frozen"]["content"]["operating_points"]["tfidf_lr"] if "content" in (m.get("frozen") or {}) else None
    sv, f, tg = g["systems"], g.get("filter") or {}, ph["tfidf_gaming"]["sets"]
    hl = ph["holm"]
    v1n, v1c, v2n = sv["v1"]["fake_case_note"], sv["v1"]["neutral"], sv["v2"]["fake_case_note"]
    gm, fr = ph["guarded_mechanism"], ph["filter"]
    dvh = ph["dev_vs_heldout"]
    nz, rt = t["noise"], m.get("retest") or {}
    swap = max(x["swap_used_mb_after"] for x in c["concurrency"]) / 1024 if c.get("concurrency") else None
    held_h = {x["version"]: 1000 * x["hours"] / x["calls"] for x in c["by_run"] if x["set"] == "test"}
    rows_e = {x["system"]: x for x in (m.get("estimates") or {}).get("rows", [])}
    est = rows_e.get("tfidf_lr")
    llm_never = all(not (rows_e[k][w] or {}).get("n_auto") for k in ("v1", "v2", "guarded") if k in rows_e
                    for w in ("realistic", "optimistic", "best_on_scored_split"))
    sub = tg["gaming_subset"]
    ids = sub["rows"]["names_towards_relief"]
    alld = tg["all_dev"]["rows"]["names_towards_relief"]
    others = [sub["rows"][a]["to_relief"] for a in g["attacks"]]
    out = [
        "Only the claims above were frozen before the run. Everything in this section is exploratory: the analyses "
        "and their wording were chosen after the held-out results were seen, and the bullets that restate a frozen "
        "claim say so.",
        "",
        f"- **Which way the errors go.** Payments the company did not make / refusals where it paid, held-out: " +
        "; ".join(f"{NAME[k]} {_errors(r[k])}" for k in ("v1", "v2", "guarded", "tfidf_lr", "issuer_rate"))
        + f". No money figure is given: the records hold {', '.join(ph['record_fields'])} and no amount, and the "
        "dollar figures in the narratives are the consumer's own, unverified and not the refunded amount.",
        f"- **The issuer.** The issuer lookup's held-out kappa {R.ci(r['issuer_rate']['kappa'])} is the highest point "
        f"estimate, but TF-IDF cannot be told apart from it (kappa difference {_sg(h['tfidf_minus_issuer_kappa'])}, "
        f"AUC difference {_sg(df['issuer_rate_minus_tfidf_lr']['auc'])}), and its threshold (the TRAIN base rate) was "
        f"chosen after DEV; the planned rule, which thresholds the same rates at one half, gets "
        f"{R.num(r['company_rule']['kappa']['value'])}. Masking issuer names lowers TF-IDF's AUC "
        f"({_sg(df['tfidf_masked_minus_tfidf_lr']['auc'])}). Within one issuer the text systems reach AUC "
        f"{R.num(min(x['value'] for x in wi.values()), 2)} to {R.num(max(x['value'] for x in wi.values()), 2)}; the "
        f"highest lower bound is {R.num(wi[top]['lo'])}, for {NAME[top]}" +
        (f" ({R.num(w10['lo'])} with {w10['B']:,} resamples instead of {ph['auc_bootstrap_b']:,})" if top == "v2" else "") +
        f", so no text system is clearly better than chance within an issuer on held-out; on DEV, TF-IDF's within-issuer AUC was "
        f"{R.ci(dv)}, clear of 0.5. The lookup encodes each issuer's past habit: it shows what the label measures and "
        "is not a policy to deploy; inside one company it is a constant.",
        f"- **What the LLM adds over a lookup or TF-IDF** (restates the frozen claims): nothing this sample can show. "
        f"In {len(h['bonferroni_judge_vs_baseline'])} paired comparisons the judges are below in "
        f"{sum(1 for c in t['claims'] if c['id'] == 'judge_vs_baseline' and _claim_status(c) == 'decided')} and "
        "not distinguishable in the rest. This is a negative result for these two prompts on one local model, not for "
        "LLMs on this task.",
        f"- **The automation slice is mostly an issuer lookup.** TF-IDF at its frozen threshold automated "
        f"{at['n_auto']}/{at['n']} at {R.pct(at['agreement']['value'])} ({_wilson(at)}; frozen claim: not shown to meet "
        f"the target). In {sl['same_as_issuer_lookup']} of those {sl['n_auto']} the decision is the one the issuer lookup "
        f"makes; {sl['narrative_names_issuer']} of the {sl['n_auto']} narratives name an issuer; the proposed refusals "
        f"come from issuers that paid {R.pct(sl['issuer_rate_auto_denied'], 0)} of their TRAIN complaints on average, the "
        f"proposed refunds from issuers that paid {R.pct(sl['issuer_rate_auto_refunded'], 0)}. The masked model makes the "
        f"same decision in {sl['masked_same_decision']} of the {sl['n_auto']}, so issuer habit shows in the wording as "
        f"well as in the names. The DEV operating point qualified on the boundary " +
        (f"({op['dev_automated']} cases, the minimum, at {R.pct(op['dev_agreement_on_automated'])}, the target) " if op else "") +
        f"and {sl['within_002_of_threshold']} held-out cases lie within 0.02 of the threshold. Inside a single company "
        "the issuer is fixed, so this slice would not exist in this form, and within an issuer no text system is "
        f"clearly above chance. Routing as measured: decide alone, none; for a person to approve, {at['auto_no_relief']} "
        f"proposed refusals (the company paid {at['wrong_no_relief']}) and {at['auto_relief']} proposed refunds (it paid "
        f"{at['auto_relief'] - at['wrong_relief']}); escalate, {at['n'] - at['n_auto']}. Whether pre-filled proposals "
        "save a reviewer time was not measured.",
        f"- **Gaming the judges** (a fixed set of {len(g['attacks'])} appended English sentences, one template each, on "
        f"{g['n_items']} DEV complaints, plus two controls; a smoke test, not a red team). The fake internal note turned "
        f"{v2n['to_relief']} of {v2n['n_base_no_relief']} of judge v2's refusals and {v1n['to_relief']} of "
        f"{v1n['n_base_no_relief']} of judge v1's into refunds (the neutral sentence: {v1c['to_relief']} of "
        f"{v1c['n_base_no_relief']} for v1); with {v1n['n_base_no_relief']} refusals the test cannot tell v1 from the "
        f"control: it shows that v2 is moved and cannot show whether v1 is. Of the {hl['tests']} rewrite-against-control tests for the "
        f"two judges, {len(hl['survive'])} survives a Holm correction ({', '.join(x.replace('fake_case_note', 'fake note') for x in hl['survive'])}). "
        f"The input filter is seven English patterns: it caught the one wording it was written against "
        f"({f['attacked_flagged']['injection']}/{g['n_items']} copies), {f['paraphrases_flagged']} of "
        f"{f['paraphrases']} other wordings of the same instruction, {fr['further_flagged']} of {fr['further_wordings']} "
        f"obfuscated wordings written after the review, {f['attacked_flagged']['fake_case_note']} of {g['n_items']} fake "
        f"notes, and it flags {fr['benign_flagged']} of {fr['benign']} harmless sentences, so its flag is a routing "
        f"lever the complainant holds. The guarded pipeline flipped {sv['guarded']['fake_case_note']['flips']}/"
        f"{g['n_items']} on the note, but it pays {sv['guarded']['base_relief']} of these {g['n_items']} and "
        f"{m['dev']['guarded']['confusion']['relief_relief'] + m['dev']['guarded']['confusion']['none_relief']} of "
        f"{m['dev']['guarded']['n']} DEV complaints, and a rule that always refuses would show no flips at all, so a "
        "low flip rate alone shows little. The note did change "
        f"the facts v2 extracted (issuer error false to true in {gm['error_changes'].get('False -> True', 0)} of "
        f"{gm['n']}, issuer status changed in {sum(gm['status_changes'].values())}), and the fact model happens to "
        "weight 'credited' and 'pending' towards no relief; the low flip rate is not shown robustness.",
        f"- **Gaming TF-IDF** (DEV only, no model call). A sentence naming four issuers whose names TF-IDF weights "
        f"towards relief turned {ids['to_relief']} of {ids['n_denials']} of its refusals on the same {sub['n']} "
        f"complaints into refunds ({R.pct(ids['to_relief_ci'][0], 0)} to {R.pct(ids['to_relief_ci'][1], 0)}, Wilson), "
        f"and {alld['to_relief']} of {alld['n_denials']} on all {tg['all_dev']['n']} DEV complaints; the neutral "
        f"sentence turned {sub['rows']['neutral']['to_relief']}, the five rewrites {min(others)} to {max(others)}, and "
        f"naming four issuers weighted towards no relief turned "
        f"{sub['rows']['names_towards_no_relief']['to_no_relief']} of {sub['rows']['names_towards_no_relief']['n_relief']} "
        "refunds into refusals. The sentences were written with the published feature list in view.",
        "- **Not tested by the gaming test**: " + "; ".join(GAPS) + ". The prompts, cues and fact-model weights are "
        "public in this repository. Text written by the claimant is evidence to check against the system of record, "
        "not an instruction.",
        f"- **DEV against held-out.** Judge v2 moved from {R.num(m['dev']['v2']['kappa']['value'])} to "
        f"{R.num(r['v2']['kappa']['value'])}: {_sg(dvh['v2'])} as two independent samples, within sampling variation "
        f"(the change in v2 minus v1 is {_sg(dvh['v2_minus_v1_change'])}); no DEV-to-held-out test was registered. The "
        f"guarded pipeline moved by {_sg(dvh['guarded'])}, which is not ordinary noise; its DEV figure is out-of-fold "
        "(ten fact models, each fitted on nine tenths of DEV) while the held-out figure uses one model fitted on all "
        "of DEV, a different estimator.",
        "- **Repeatability** (the noise pass was frozen; this reading was not). Asked twice, the judges gave the same "
        "decision in " + " and ".join(f"{x['same_decision']}/{x['n']}" for x in nz.values()) + " held-out cases and "
        + " and ".join(f"{x['same_decision']}/{x['n']}" for x in rt.values()) + f" DEV cases ({', '.join(nz)}). On held-out, judge v2's "
        f"answers were byte-identical in {nz['v2']['byte_identical']}/{nz['v2']['n']}, against "
        f"{rt['v2']['byte_identical']}/{rt['v2']['n']} on DEV, where its confidence moved by up to "
        f"{R.num(rt['v2']['shift']['max'], 2)}" + (f" and the laptop ran with up to {swap:.0f} GB in swap" if swap else "")
        + ": the held-out pass measured a quiet machine. The confidence score is not stable enough to set a threshold "
        "on.",
        f"- **Cost** (compute on one Apple M4 Max laptop with 36 GB of memory and a local model, DEV timings; no money "
        f"cost or handle time was measured): a table lookup for the issuer rows; TF-IDF "
        f"{c['tfidf_lr']['predict_ms_per_item_one_by_one']:.1f} s of CPU per 1,000; judge v1 {c['v1']['hours_per_1000']:.1f} h, "
        f"judge v2 {c['v2']['hours_per_1000']:.1f} h" +
        (" (on the held-out run " + " and ".join(f"{v} {x:.1f} h" for v, x in held_h.items()) + ")" if held_h else "") +
        "; "
        "a person: " + (f"about {_people(m)[0][1]['median_seconds'] * 1000 / 3600:.0f} h at the median labelling time."
                        if _people(m) else "not measured until the blind labels are locked."),
        (f"- **Realistic against optimistic** (plan item 9; full table in the report's Estimates): TF-IDF automates "
         f"{R.pct(est['realistic']['coverage'])} at {R.pct(est['realistic']['agreement']['value'])} on held-out at the "
         f"frozen threshold, {R.pct(est['optimistic']['coverage'])} at {R.pct(est['optimistic']['agreement']['value'])} on "
         f"DEV where the threshold was chosen, and {R.pct(est['best_on_scored_split']['coverage'])} at "
         f"{R.pct(est['best_on_scored_split']['agreement']['value'])} with the best threshold chosen on the held-out split "
         "itself" + ("; the LLM configurations automate nothing in any of the three." if llm_never else ".")
         if est and est["realistic"] and
         est["optimistic"] and est["best_on_scored_split"] else "- **Realistic against optimistic**: see the report."),
        "- **Not tried**: few-shot examples, giving the LLM the issuer's payout rate, a larger or hosted model, a label "
        "written by trained agents under a written policy. **Not measured**: handle time, money, customer outcomes, "
        "appeal rate, chargeback reason codes and scheme deadlines, billing-error time limits, who is accountable, "
        "whether a reviewer can use the output.",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ exploratory detail (report only)
def b_exploratory(m):
    if not _ready(m):
        return R.PENDING
    t, ph = m["test"], m["posthoc"]
    r, h = t["results"], ph["heldout"]
    out = [f"Computed by `src/judge/posthoc.py` after the held-out run, from the same data and caches; nothing here "
           f"was registered and none of it decides a claim. Kappa and accuracy intervals: {m['settings']['bootstrap_b']:,} "
           f"percentile-bootstrap resamples over complaints (paired where two systems are compared; DEV and held-out "
           f"resampled independently when they are compared); AUC intervals elsewhere in this report use "
           f"{ph['auc_bootstrap_b']:,} resamples, as the frozen code does.\n",
           "Held-out, by system: agreement with the record against always answering no relief (paired), and the "
           "direction of the errors:\n",
           R.table(("System", "Matches the record", "Minus always no relief [95% CI]", "Would pay",
                    "Pays where the company did not", "Refuses where the company paid"),
                   [(SHORT[k], R.pct(r[k]["accuracy"]["value"]), _pts(x), _pays(r[k]), r[k]["confusion"]["none_relief"],
                     r[k]["confusion"]["relief_none"]) for k, x in h["accuracy_minus_always_no"].items()]),
           f"\nThe records hold {', '.join(ph['record_fields'])}; there is no amount field, so no error is given in money.",
           f"\nTF-IDF minus the issuer relief rate, held-out kappa (paired): {_sg(h['tfidf_minus_issuer_kappa'])}.",
           "\nThe six frozen judge-against-baseline differences at the Bonferroni level for six comparisons "
           f"({R.pct(next(iter(h['bonferroni_judge_vs_baseline'].values()))['level'], 2)} intervals, same resamples): " +
           "; ".join(f"{k.replace('_minus_', ' minus ')} {_sg(x)}" for k, x in h["bonferroni_judge_vs_baseline"].items()) + ".",
           "\nDEV against held-out kappa (independent samples; two-sided bootstrap p): " + "; ".join(
               f"{k.replace('_minus_v1_change', ' minus v1, change')} {_sg(x)} (p {x['p_two_sided']:.2f})"
               for k, x in ph["dev_vs_heldout"].items()) + ".",
           f"\nJudge v2's within-issuer AUC on held-out with {h['v2_within_issuer_at_10000']['B']:,} resamples: "
           f"[{R.num(h['v2_within_issuer_at_10000']['lo'])}, {R.num(h['v2_within_issuer_at_10000']['hi'])}], "
           f"{R.pct(h['v2_within_issuer_at_10000']['share_at_or_below_half'])} of resamples at or below 0.5. The "
           "within-issuer intervals resample complaints, not issuers; resampling issuers would give wider ones.",
           f"\nThe TF-IDF slice automated at the frozen threshold ({h['tfidf_slice']['n_auto']} cases from "
           f"{h['tfidf_slice']['issuers']} issuers, the largest contributing {R.pct(h['tfidf_slice']['largest_issuer_share'], 0)}): "
           f"same decision as the issuer lookup in {h['tfidf_slice']['same_as_issuer_lookup']}, as masked TF-IDF in "
           f"{h['tfidf_slice']['masked_same_decision']}; narratives naming an issuer {h['tfidf_slice']['narrative_names_issuer']}; "
           f"held-out cases within 0.02 of the threshold {h['tfidf_slice']['within_002_of_threshold']}.",
           "\nTRAIN relief rate by the sub-issue the consumer selected:\n",
           R.table(("Sub-issue", "TRAIN complaints", "Closed with monetary relief"),
                   [(k, f"{v['n']:,}", R.pct(v["relief_rate"])) for k, v in ph["sub_issues"].items()])]
    tg = ph["tfidf_gaming"]
    out.append("\nGaming the TF-IDF baseline: each sentence appended to the narrative and scored by the TF-IDF model "
               "(fitted on TRAIN with the frozen settings). Refusals turned into refunds [95% Wilson CI], refunds "
               "turned into refusals, mean change in P(relief), and the exact McNemar comparison with the neutral "
               "control on refusals turned into refunds:\n")
    for name, s in tg["sets"].items():
        label = "the gaming subset" if name == "gaming_subset" else "all of DEV"
        out.append(f"\n{label.capitalize()} ({s['n']} complaints, {s['base_denials']} refusals on the unmodified text):\n")
        out.append(R.table(("Sentence", "Refusals turned into refunds", "Refunds turned into refusals",
                            "Mean change in P(relief)", "Against the neutral control"),
                           [(ATTACK_WORDS.get(a, a), f"{x['to_relief']}/{x['n_denials']} [{R.pct(x['to_relief_ci'][0])}, "
                             f"{R.pct(x['to_relief_ci'][1])}]", f"{x['to_no_relief']}/{x['n_relief']}",
                             f"{x['mean_delta_p']:+.3f}",
                             (f"{x['vs_neutral']['only_sentence']} / {x['vs_neutral']['only_control']}, p " +
                              ("below 0.001" if x["vs_neutral"]["p"] < 0.001 else f"{x['vs_neutral']['p']:.3f}"))
                             if "vs_neutral" in x else "")
                            for a, x in s["rows"].items()]))
    out.append("\nThe issuer-name sentences: " + "; ".join(f"{ATTACK_WORDS[k]}: \"{tg['sentences'][k]}\""
                                                           for k in ("names_towards_relief", "names_towards_no_relief")) + ".")
    fr, gm, hl = ph["filter"], ph["guarded_mechanism"], ph["holm"]
    out.append(f"\nInput filter on further wordings (written after the review, checked against the filter only): "
               f"{fr['further_flagged']} of {fr['further_wordings']} obfuscated or reworded instructions flagged, "
               f"{fr['benign_flagged']} of {fr['benign']} harmless sentences flagged, {fr['heldout_flagged']} of "
               f"{fr['heldout_n']} held-out narratives flagged (`FURTHER_WORDINGS` and `BENIGN` in `src/judge/posthoc.py`).")
    out.append(f"\nWhat the fake internal note changed in judge v2's extracted facts on the {gm['n']} gaming complaints: "
               "issuer status " + ", ".join(f"{k} {v}" for k, v in gm["status_changes"].items()) + "; issuer error " +
               ", ".join(f"{k} {v}" for k, v in gm["error_changes"].items()) + ". Fact-model weights (DEV fit): " +
               ", ".join(f"{k} {v:+.2f}" for k, v in gm["weights"].items()) + ".")
    small = ", ".join(f"{x['system']} {x['rewrite']} " + ("below 0.001" if x["p"] < 0.001 else f"{x['p']:.4f}")
                      for x in hl["smallest"])
    out.append(f"\nHolm correction over the {hl['tests']} rewrite-against-control tests of the two judges: surviving "
               f"{', '.join(hl['survive']) or 'none'}; the smallest p-values {small}.")
    out.append("\nKnown gaps of the gaming test: " + "; ".join(GAPS) + ".")
    return "\n".join(out)


# ------------------------------------------------------------------ the freeze record
def b_freeze_record(m):
    ph = m.get("posthoc")
    if not ph:
        return R.PENDING
    gi, t = ph["integrity"]["gist"], m.get("test") or {}
    rows = [(x["committed_at"].replace("T", " ").replace("Z", " UTC"), f"`{x['commit'][:12]}`",
             f"`{x['frozen_sha256'][:16]}...`") for x in gi["revisions"]]
    ok = not ph["integrity"]["problems"]
    from datetime import datetime
    gap = "n/a"
    if t.get("started_at"):
        sec = (datetime.fromisoformat(t["started_at"]) -
               datetime.fromisoformat(gi["revisions"][-1]["committed_at"].replace("Z", "+00:00"))).total_seconds()
        gap = f"{int(sec // 60)} minutes {int(sec % 60)} seconds"
    return "\n".join([
        f"Gist: {gi['url']} (secret, that is unlisted: readable by anyone with the link). Revisions, with the "
        "commit time GitHub's API reports for each:\n",
        R.table(("Revision committed (GitHub server time)", "Frozen commit it names", "SHA-256 of results/frozen.json"),
                rows),
        f"\nThe run marker (`results/test_run.json`) records the start of the held-out run at "
        f"{t.get('started_at', 'n/a').replace('T', ' ').replace('+00:00', ' UTC')}, from commit "
        f"`{t.get('commit', '')[:12]}`; the frozen file now on disk has SHA-256 `{(m.get('frozen') or {}).get('sha256', '')[:16]}...`. "
        + ("Both match the gist's last revision (`make check` compares them with the hashes copied from it). " if ok else
           "They do not match the gist's last revision. ") +
        "The first revision names an earlier freeze, replaced before any held-out complaint was sent to the model.",
        f"\nWhat this shows: by the time of the last revision, {gap} before the run marker, the holder of the "
        "author's GitHub account had the frozen commit hash and the hash of the frozen file, and a commit hash fixes "
        "every file of that commit. What it does not show: that no held-out complaint had been sent to the model "
        "before then (that rests on the request journal and the caches in this repository), or anything about the "
        "times of other commits. A gist is controlled by its owner and is not a notarised record; its revision times "
        "are GitHub's, not an independent timestamping service's.",
        f"\nAnswer files at the run commit `{ph['integrity']['run_commit']}`, recorded after the run (they were not in "
        "the run marker): " + "; ".join(f"`{k}` {v['bytes']:,} bytes, SHA-256 `{v['sha256'][:16]}...`"
                                        for k, v in ph["integrity"]["run_files"].items()) +
        ". `make check` fails if these bytes change; they anchor later changes only once that commit is public."])


# ------------------------------------------------------------------ the blind labels
def _head(h) -> str:
    t = h["translations"]
    return (f"Sample: {h['sample_n']} {h['split'].upper()} complaints drawn by a hash of the ID "
            "(`data/label_sample.json`), which no model had answered and the annotator had not seen; labelled blind by "
            "the author against `docs/labelling_rubric.md`, which gives the person the rules the judge v1 prompt was "
            "given. Person against v1 therefore measures whether the model applies a written policy the way a person "
            "reading the same text does; person against v2 compares answers to different questions. The labels are one "
            "non-expert's reading through a machine translation, not ground truth. The analysis was registered before "
            f"any label existed (`docs/analysis_plan_human_labels.md`, {h['plan_registered_at'][:16].replace('T', ' ')} UTC, "
            f"amended {h['amended_at'][:16].replace('T', ' ')} UTC). The Russian reading aid beside the English original "
            "was made by a separate model, not the judged one (Anthropic's `claude-fable-5-2`, run as sub-agents of the "
            "AI coding assistant that wrote this repository, so those sub-agents read the 100 narratives), never part of "
            f"a judged input ({t['flagged']} translations fail a completeness check, {t['missing']} are missing).")


NOT_REGISTERED = "not in the registered plan; descriptive"


def _against_person(f) -> str:
    rows = []
    for n, x in f["systems_vs_person"].items():
        if n not in SHORT:
            continue
        c = x["confusion"]          # here the person's label is the reference
        note = {"v1": " (same question as the rubric)", "v2": " (a different question)"}.get(n, "")
        rows.append((SHORT[n] + note, R.ci(x["kappa"]), R.pct(x["accuracy"]["value"]),
                     R.ci(f["sensitivity"]["systems_vs_person"][n]["kappa"]), R.pct(x["predicted_positive_rate"]),
                     f"{c['none_relief']} / {c['relief_none']}"))
    return R.table(("System against the person", "Kappa [95% CI], decided cases", "Matches the person",
                    "Kappa with 'cannot tell' as no money back", f"System pays ({NOT_REGISTERED})",
                    f"System pays, person would not / person would pay, system would not ({NOT_REGISTERED})"), rows)


def _v3(m) -> str:
    from .prompts import VERSIONS
    if "v3" not in VERSIONS:
        return ("No v3 prompt exists. If one is written, it is drafted from the calibration half only, frozen, run "
                "once on the evaluation half, and accepted only if the paired interval of kappa against the person "
                "(v3 minus the better of v1 and v2 on the calibration half) lies entirely above zero on the evaluation "
                "half, as registered; otherwise the report says that no improvement was demonstrated.")
    return "A v3 prompt exists; its evaluation-half result is reported only through the registered rule."


def b_human(m):
    h = m["human"]
    out = [_head(h)]
    if not h["annotators"]:
        out.append(f"\n_No locked set of labels yet._ Nothing is reported from a partial set. Model answers for the "
                   f"sampled complaints so far: {h['model_answers_for_sample']} (the judges run on them only after the "
                   "lock, with `make judge-labelled`).")
        return "\n".join(out)
    extra = (m.get("posthoc") or {}).get("human", {})
    for who, r in h["annotators"].items():
        f, s = r["all"], extra.get(who, {}).get("seconds")
        out.append(f"\nAnnotator `{who}` (the author; not a lawyer; no second reader, so no inter-annotator agreement), "
                   f"locked {r['locked_at'][:16].replace('T', ' ')} UTC: {f['decided']} decided "
                   f"({R.pct(f['person_relief_share'])} of them money back; the companies paid in "
                   f"{R.pct(f['recorded_relief_share'])} of the sample), {f['cannot_tell']} cannot tell "
                   f"({R.pct(f['cannot_tell'] / f['n'], 0)}); time per label: median {r['median_seconds']:.0f} s" +
                   (f", quartiles {s['p25']:.0f} s and {s['p75']:.0f} s, {s['under_30']} under 30 s" if s else "") +
                   f", {r['total_minutes']:.0f} minutes in all.")
        if "person_vs_recorded" not in f:
            out.append("\nToo few decided cases for agreement figures.")
            continue
        out.append("\n" + person_table({"human": {"annotators": {who: r}}}) +
                   f"\n\n(The person's row is registered; the system rows of that table are {NOT_REGISTERED}.)")
        if not any(v in f["systems_vs_person"] for v in ("v1", "v2")):
            out.append("\nThe judges have not answered these complaints yet (`make judge-labelled`).")
        out.append("\n" + _against_person(f))
        p = f["paired_vs_person"]
        if p:
            out.append("\nPaired differences in kappa against the person (registered): " + "; ".join(
                f"{SHORT[k.split('_minus_')[0]]} minus {SHORT[k.split('_minus_')[1]]} {_sg(x['kappa'])}"
                for k, x in p.items()) + ".")
    out.append("\n" + _v3(m) + " The author's own reading of each sorted disagreement, from the second screen, is in "
               "the full report.")
    return "\n".join(out)


def b_human_detail(m):
    h = m["human"]
    if not h["annotators"]:
        return "_No locked set of labels yet._"
    extra = (m.get("posthoc") or {}).get("human", {})
    out = []
    for who, r in h["annotators"].items():
        f = r["all"]
        out.append(f"Annotator `{who}`. Cases answered 'cannot tell': {f['cannot_tell']}; recorded relief rate among "
                   f"them {R.pct(f['cannot_tell_recorded_relief'])}; relief rate of each system on them: " +
                   ", ".join(f"{SHORT.get(n, n)} {R.pct(v)}" for n, v in f["cannot_tell_system_relief"].items()) +
                   f". Notes written: {r['with_note']}. Cases shown with a flagged translation: "
                   f"{r['translation_flagged_when_shown']}" +
                   (f"; translation shown in {extra[who]['translation_shown']} of {f['n']}." if who in extra else "."))
        for half, g in r["halves"].items():
            out.append(f"\n{half.capitalize()} half ({g['n']} cases, {g['decided']} decided, {g['cannot_tell']} cannot "
                       "tell); the registered plan reads these only through the v3 rule:\n")
            out.append(_against_person(g) if "person_vs_recorded" in g else "Too few decided cases.")
        e = r["error_review"]
        out.append(f"\nSecond screen (after the lock), the author's own reading of the {e['total']} individually sorted "
                   f"v1 DEV disagreements: {e['reviewed']} reviewed, {e['confirmed']} drafted readings confirmed, "
                   f"{len(e['changed'])} changed.")
        rows = (extra.get(who) or {}).get("review") or []
        if rows:
            out.append("\n" + R.table(("Complaint (DEV)", "Drafted reading (AI coding assistant)", "Author's reading",
                                       "Author's note"),
                                      [(x["complaint_id"], x["drafted"].replace("_", " "),
                                        ("same" if x["confirmed"] else x["author"].replace("_", " ")), x["note"])
                                       for x in rows]))
    return "\n".join(out)
