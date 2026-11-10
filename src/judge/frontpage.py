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
NOT_REGISTERED = "not in the registered plan; descriptive"
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


# The commits that wrote results/frozen.json, oldest first (from `git log -- results/frozen.json`; copied here
# so that the check runs without git history). The first was removed again before the second was made.
FREEZES = ("f269005", "3f97ae2", "97dd167", "822811d")


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
def _kappa_leaders(r) -> list[str]:
    names = [k for k in SHORT if k != "majority" and k in r]
    return sorted(names, key=lambda k: -r[k]["kappa"]["value"])[:2]


def _within_issuer_range(m) -> tuple[float, float]:
    sy = m["test"]["ranking"]["systems"]
    v = [sy[k]["within_issuer_auc"]["value"] for k in TEXT_SYSTEMS if k in sy]
    return min(v), max(v)


def b_first_screen(m):
    if not _ready(m):
        return R.PENDING
    t, ph = m["test"], m["posthoc"]
    r, n = t["results"], t["n"]
    ex = ph["accuracy_vs_always_no_exact"]
    accs = [r[k]["accuracy"]["value"] for k in ex["systems"]]
    clearly = [k for k, x in ex["systems"].items() if x["lo"] > 0]
    lead = _kappa_leaders(r)
    _, share = _issuer_share(m)
    pdn = m["error_analysis"]["prior_denial"]
    proxy = next(x for x in m["cues"]["rows"] if x["cue"] == "issuer had denied the dispute")
    text_best = max(("tfidf_lr", "v1", "v2", "guarded"), key=lambda k: r[k]["kappa"]["value"])
    df = t["ranking"]["differences"]
    lo_w, hi_w = _within_issuer_range(m)
    wb = ph["within_issuer_by_issuer"]
    v2n = m["gaming"]["systems"]["v2"]["fake_case_note"]
    tg = ph["tfidf_gaming"]["sets"]["gaming_subset"]["rows"]["names_towards_relief"]
    at = t["automation"]["tfidf_lr"]["at_dev_threshold"]
    toward_pay = [k for k in ("v2", "guarded") if r[k]["confusion"]["none_relief"] < r[k]["confusion"]["relief_none"]]
    out = [
        "*Here the \"judge\" is the model that makes the refund decision, not a model that grades other models' "
        "answers.*",
        "",
        f"**The data.** Complaints that US consumers sent to the regulator (the CFPB) about a disputed credit-card "
        f"charge. Most came after the card issuer had already handled the dispute: {R.pct(share, 0)} of the "
        f"{m['base_rates']['train']['n']:,} training complaints are filed as 'company isn't resolving a dispute', and "
        f"in {pdn['n_denied']} of {m['error_analysis']['n_dev']} development complaints the second prompt (judge v2) "
        f"records that the issuer had already refused (a cruder keyword search finds a refusal in "
        f"{R.pct(proxy['n_with'] / m['cues']['n_train'], 0)} of the training complaints). The label is how the "
        "company closed the complaint with the regulator, with money back (goodwill credits included) or without, so "
        "every figure here is agreement with what the company did, not correctness." + _person_first_screen(m),
        "",
        f"- **Always answering \"no money back\" matches the record in {R.pct(r['majority']['accuracy']['value'])} of "
        f"the {n} held-out complaints" +
        (", and no system is clearly above it.**" if not clearly else ".**") +
        f" {EXPL} The {len(accs)} systems match the record in {R.pct(min(accs))} to {R.pct(max(accs))}. Agreement flatters systems "
        "that mostly say no, so the plan scores them by kappa (agreement beyond what the base rate gives), where the "
        + " and the ".join(f"{NAME[k]} ({R.num(r[k]['kappa']['value'])})" for k in lead) + " lead.",
        f"- **A model applying the written refund rules to the consumer's account (judge v1) would pay "
        f"{_pays(r['v1'])} of the {n}; the companies paid {r['majority']['confusion']['relief_none']}.** {EXPL} It errs "
        f"towards paying ({r['v1']['confusion']['none_relief']} payments the company did not make, "
        f"{r['v1']['confusion']['relief_none']} refusals where it paid)" +
        (f"; {' and '.join(('the ' if k == 'guarded' else '') + NAME[k] for k in toward_pay)} err towards refusing "
         f"({'; '.join(_errors(r[k]) for k in toward_pay)})." if toward_pay else "."),
        f"- **The record follows which issuer answered** {EXPL}: knowing the bank does about as well as reading "
        "the complaint. A lookup of each issuer's past payout rate, which never reads the text, agrees with the record "
        f"about as well as the best text model (kappa {R.num(r['issuer_rate']['kappa']['value'])} against "
        f"{R.num(r[text_best]['kappa']['value'])}, difference {_sg(ph['heldout']['tfidf_minus_issuer_kappa'])}). Hiding "
        f"the issuer's name from the text model makes it rank complaints worse (AUC "
        f"{_sg(df['tfidf_masked_minus_tfidf_lr']['auc'])}). Among complaints about the same issuer the text systems "
        f"rank paid above unpaid only weakly (AUC {R.num(lo_w, 2)} to {R.num(hi_w, 2)}); whether that beats a coin "
        f"toss depends on whether complaints or issuers are resampled, and only {wb['issuers_with_pairs']} issuers "
        "carry those pairs.",
        f"- **Decision: automate nothing.** The plan required {R.pct(m['settings']['target_agreement'], 0)} agreement "
        "on the cases a system would decide alone, and no system is shown to reach it. The only system with a frozen "
        f"threshold, a classic text classifier (TF-IDF), would decide {at['n_auto']} of the {at['n']} alone and agrees "
        f"on {R.pct(at['agreement']['value'])} of them, but on so few cases the interval runs from "
        f"{R.pct(at['agreement_wilson'][0])} to {R.pct(at['agreement_wilson'][1])} (Wilson). In a test on development "
        f"complaints a fake internal note pasted into the text turned {v2n['to_relief']} of "
        f"{v2n['n_base_no_relief']} of judge v2's refusals into refunds, and a sentence naming four banks turned "
        f"{tg['to_relief']} of {tg['n_denials']} of TF-IDF's refusals on the same complaints into refunds.",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ frozen claims and the table
def _claims(m) -> str:
    """Each frozen claim with the verdict words its rule in results/frozen.json produced, verbatim."""
    t, st, ph = m["test"], m["settings"], m["posthoc"]
    cl = t["claims"]
    out = []
    for c in cl:
        if c["id"] == "v2_vs_v1":
            out.append(f"- *v2 replaces v1* (only if the paired kappa difference, v2 minus v1, lies entirely above "
                       f"zero): **{c['verdict']}**, {_sg(c['interval'])}.")
    jb = [c for c in cl if c["id"] == "judge_vs_baseline"]
    dec = [c for c in jb if _claim_status(c) == "decided"]
    und = [c for c in jb if _claim_status(c) != "decided"]
    bon = ph["heldout"]["bonferroni_judge_vs_baseline"]
    held = [k for k in bon if bon[k]["hi"] < 0]
    out.append(f"- *Each judge against TF-IDF, the issuer relief rate and the company rule* ({len(jb)} paired kappa "
               "differences, judge minus baseline): " + "; ".join(f"**{c['verdict']}** ({_sg(c['interval'])})"
                                                               for c in dec) +
               (f"; both also exclude zero at the {R.pct(bon[held[0]]['level'], 2)} level that corrects for "
                f"{len(bon)} comparisons {EXPL}" if len(held) == len(dec) and dec else "") +
               ("; " if dec and und else "") +
               "; ".join(f"**{c['verdict']}** ({_sg(c['interval'])})" for c in und) + ".")
    wi = [c for c in cl if c["id"] == "within_issuer"]
    out.append("- *TF-IDF orders complaints better than the judge within one issuer* (AUC difference, TF-IDF minus "
               "the judge): " + "; ".join(f"**{c['verdict']}** ({_sg(c['interval'])})" for c in wi) + ".")
    at = t["automation"]["tfidf_lr"]["at_dev_threshold"]
    for c in cl:
        if c["id"] == "tfidf_operating_point":
            out.append(f"- *The TF-IDF threshold frozen on DEV meets the {R.pct(st['target_agreement'], 0)} agreement "
                       f"target on held-out cases*: **{c['verdict']}** "
                       f"({at['n_auto']}/{at['n']} automated, {R.pct(at['agreement']['value'])}, {_wilson(at)}; "
                       f"always answering no relief agrees on {R.pct(at['majority_agreement_on_auto'])} of the same "
                       "cases).")
    nz = t["noise"]
    llm = next((c["verdict"] for c in cl if c["id"] == "llm_automation"), None)
    out.append("- Two frozen items record facts of the procedure rather than test a result, so they could not come "
               "out otherwise; they are listed so that the list is complete: " +
               (f"**{llm}**, because none of them had a DEV threshold; " if llm else "") +
               "and the noise pass was done (" + ", ".join(
                   f"judge {v} repeated {r['same_decision']}/{r['n']} decisions" for v, r in nz.items()) + ").")
    return "\n".join(out)


def _table(m) -> str:
    t, d, b = m["test"], m["dev"], m["base_rates"]
    rows = []
    for name in R.SUMMARY_ROWS:
        if name not in d or name not in t["results"]:
            continue
        r = t["results"][name]
        rows.append((SHORT[name], READS[name], R.pct(r["accuracy"]["value"]),
                     f"{_pays(r)} ({r['confusion']['none_relief']})", r["confusion"]["relief_none"],
                     R.kappa_cell(r), R.kappa_cell(d[name]), _automated_cell(t, name), R.time_per_1000(m, name)))
    paid = t["results"]["majority"]["confusion"]["relief_none"]
    return (R.table(("System", "Reads", f"Matches the record, held-out (n={b['test']['n']}) {EXPL}",
                     f"Pays, of {b['test']['n']} (of these, the company did not pay) {EXPL}",
                     f"Refuses where the company paid (it paid {paid}) {EXPL}",
                     "Held-out kappa [95% CI]", f"DEV kappa [95% CI], n={b['dev']['n']}",
                     "Automated on held-out at the frozen threshold", "Time per 1,000 decisions (DEV timings)"), rows) +
            f"\n\nKappa is agreement beyond what the base rate gives (0 for a constant answer). Relief base rate "
            f"{R.pct(b['test']['base_rate'])} held-out, {R.pct(b['dev']['base_rate'])} DEV. Text: the product and "
            "issue fields and the consumer's narrative; issuer only: the issuer's relief rate on the "
            f"{b['train']['n']:,} TRAIN complaints. DEV was used for every choice after the first prompt, so the DEV "
            "column is optimistic. The three columns marked exploratory were added after the run.")


def _people(m):
    return [(who, r) for who, r in m["human"]["annotators"].items() if "person_vs_recorded" in r["all"]]


def person_table(m) -> str:
    """The person against the recorded decision next to every system, on the same decided labelled cases.
    Never mixed into a DEV or held-out column: the labelled sample is drawn from TRAIN. The person's row is
    registered (with both readings of 'cannot tell'); the system rows are not, and say so wherever this prints."""
    out = []
    for who, r in _people(m):
        f = r["all"]
        same = f["systems_vs_recorded_same_cases"]
        pv, sv = f["person_vs_recorded"], f["sensitivity"]["person_vs_recorded"]
        rows = [("Person (blind, one annotator) (registered)", R.ci(pv["kappa"]), R.ci(pv["accuracy"], R.pct),
                 R.ci(sv["kappa"]), R.ci(sv["accuracy"], R.pct), R.pct(f["person_relief_share"]))]
        rows += [(f"{SHORT[n]} ({NOT_REGISTERED})", R.ci(x["kappa"]), R.ci(x["accuracy"], R.pct), "", "",
                  R.pct(x["predicted_positive_rate"])) for n, x in same.items() if n in SHORT]
        out.append(f"Against the recorded decision on the same {f['decided']} decided cases of the {f['n']}-complaint "
                   f"TRAIN labelling sample (`{who}`; {f['cannot_tell']} answered 'cannot tell' = "
                   f"{R.pct(f['cannot_tell'] / f['n'], 0)}, left out of the first two columns and counted as no money "
                   f"back in the next two, over all {f['n']}; baselines refitted without these complaints):\n\n"
                   + R.table(("Who", "Kappa against the record [95% CI]", "Matches the record [95% CI]",
                              "Kappa, 'cannot tell' as no money back", "Matches, 'cannot tell' as no money back",
                              f"Pays ({NOT_REGISTERED})"), rows))
    return "\n\n".join(out)


def b_headline(m):
    t = m.get("test")
    if not t:
        return R.b_lead(m)
    if not m.get("posthoc"):
        return R.PENDING
    out = [f"**Held-out run**: {t['n']} complaints, scored once from commit `{t['commit'][:7]}`, the last of "
           f"{len(FREEZES)} commits that froze the method; it precedes the run in this repository's history, and a GitHub "
           "gist created before the run holds its hashes (see the freeze section below). **Claims frozen before the run**, each with the verdict its rule "
           "in `results/frozen.json` gives, in the words the frozen code prints:\n", _claims(m),
           "\n" + _table(m)]
    out.append("\n" + human_summary(m) if _people(m) else
               "\nBlind human labels are being collected; nothing is reported until the set is complete and locked.")
    return "\n".join(out)


# ------------------------------------------------------------------ the blind labels, in short (after the lock)
AFTER_LOCK = "added after the labels were locked and read; descriptive"


def _person_facts(m):
    """(annotator, figures on all labelled cases, posthoc extras) of the first locked annotator with figures."""
    who, r = _people(m)[0]
    return who, r["all"], ((m.get("posthoc") or {}).get("human") or {}).get(who, {})


def _as_weakly(f, ex) -> bool:
    """True when the person's kappa against the record lies inside the range of the eight systems both on his
    decided cases and on the held-out split: the condition for saying 'about as weakly as the systems do'."""
    k = f["person_vs_recorded"]["kappa"]["value"]
    rg = ex.get("kappa_ranges_vs_record") or {}
    return all(x and x["min"] <= k <= x["max"] for x in (rg.get("same_decided_cases"), rg.get("held_out")))


def _person_first_screen(m) -> str:
    if not _people(m):
        return ""
    _, f, ex = _person_facts(m)
    k = R.ci(f["person_vs_recorded"]["kappa"])
    return (f" A person reading only the complaint, blind (one annotator, the author), could not decide "
            f"{f['cannot_tell']} of {f['n']}; on the {f['decided']} he decided, he agreed with the record "
            + ("about as weakly as the systems do" if _as_weakly(f, ex) else "with") +
            f" (kappa {k}; registered; see below).")


def human_summary(m) -> str:
    """The registered human-label figures in a few lines for the held-out section; the full tables print
    once, in the human-label section."""
    who, f, ex = _person_facts(m)
    pv, sv = f["person_vs_recorded"]["kappa"], f["sensitivity"]["person_vs_recorded"]["kappa"]
    sp = f["systems_vs_person"]
    rg = ex.get("kappa_ranges_vs_record") or {}
    same, held = rg.get("same_decided_cases"), rg.get("held_out")
    if same and held:
        side = lambda x: "inside" if x["min"] <= pv["value"] <= x["max"] else "outside"   # noqa: E731
        where = (f"His {R.num(pv['value'])} lies {side(same)} the range of the {same['n']} systems' kappa against the "
                 f"record on the same {f['decided']} cases ({R.num(same['min'])}, {NAME[same['min_system']]}, to "
                 f"{R.num(same['max'])}, {NAME[same['max_system']]}) and {side(held)} their range on the held-out split "
                 f"({R.num(held['min'])} to {R.num(held['max'])}) ({AFTER_LOCK}); no difference between the person and "
                 "any system was tested.")
    else:
        where = ""
    agree = [n for n, x in sp.items() if x["kappa"]["lo"] > TINY]
    judges = "; ".join(f"{NAME[v]} {R.ci(sp[v]['kappa'])}" for v in ("v1", "v2") if v in sp)
    touches = [NAME[v] for v in ("v1", "v2") if v in sp and abs(sp[v]["kappa"]["lo"]) <= TINY]
    return "\n".join([
        f"**Blind human labels** (a separate analysis on {f['n']} TRAIN complaints, not the held-out split; "
        "registered in `docs/analysis_plan_human_labels.md` before any label existed; full tables under \"A person's "
        "blind labels\" below):\n",
        f"- One annotator, the author: not a lawyer, and not a native English reader, who read the English original "
        f"with a machine translation beside it. He labelled {f['n']} complaints blind and could not decide "
        f"{f['cannot_tell']} of them from the text (registered).",
        f"- On the {f['decided']} he decided, his agreement with the company's record is kappa {R.ci(pv)} (registered)" +
        (": a person reading only the complaint agrees with the record about as weakly as the systems do. "
         if _as_weakly(f, ex) else ". ") + where +
        f" With 'cannot tell' counted as no money back, over all {f['n']} (registered sensitivity line): kappa {R.ci(sv)}.",
        f"- The judges against the person, decided cases (registered): {judges}; " +
        ("no system is shown to agree with the person (no interval lies entirely above zero" +
         (f"; {' and '.join(touches)}'s reaches down to zero)." if touches else ").")
         if not agree else "the interval lies entirely above zero for " + ", ".join(NAME[n] for n in agree) + ".") +
        " " + _sign_check(_figures(f, ex.get("paired_sensitivity", {}).get("all"))),
    ])


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
    wb = ph["within_issuer_by_issuer"]
    above = [k for k, x in wb["systems"].items() if x["lo"] > 0.5]
    ex = ph["accuracy_vs_always_no_exact"]
    accs = {k: r[k]["accuracy"]["value"] for k in ex["systems"]}
    hi_acc = max(accs, key=accs.get)
    sc, smooth, h1 = ph["tfidf_slice_correct"], ph["issuer_smoothed_at_half"], ph["holm_one_sided"]
    real = ph["filter_on_real_narratives"]
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
    sp = {}
    for e in c.get("run_speed") or []:                 # every speed logged for each pass, in order
        sp.setdefault((e["label"], e["version"]), []).append(e["generation_tokens_per_s"])
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
        f"- **Agreement against always answering no.** The {len(accs)} systems match the record in "
        f"{R.pct(min(accs.values()))} to {R.pct(max(accs.values()))} of held-out complaints, always no in "
        f"{R.pct(r['majority']['accuracy']['value'])}. The highest, the {NAME[hi_acc]}, was picked after the run as the "
        f"highest of {len(accs)}: {_pts(h['accuracy_minus_always_no'][hi_acc])} at 95%, "
        f"[{100 * ex['systems'][hi_acc]['lo']:+.1f}, {100 * ex['systems'][hi_acc]['hi']:+.1f}] at the "
        f"{R.pct(ex['level'], 3)} level that corrects for {len(accs)} comparisons; it differs from always no on "
        f"{ex['systems'][hi_acc]['system_only_right'] + ex['systems'][hi_acc]['always_no_only_right']} complaints (right "
        f"on {ex['systems'][hi_acc]['system_only_right']} of them, wrong on {ex['systems'][hi_acc]['always_no_only_right']}"
        f"; exact two-sided test p {ex['systems'][hi_acc]['p_exact_two_sided']:.3f}). The planned company rule: "
        f"{_pts(h['accuracy_minus_always_no']['company_rule'])} at 95%, [{100 * ex['systems']['company_rule']['lo']:+.1f}, "
        f"{100 * ex['systems']['company_rule']['hi']:+.1f}] corrected, on "
        f"{ex['systems']['company_rule']['system_only_right'] + ex['systems']['company_rule']['always_no_only_right']} "
        f"complaints that differ (p {ex['systems']['company_rule']['p_exact_two_sided']:.3f}). With so few complaints "
        "that differ, the exact test is a safer reading than the 95% bootstrap interval.",
        f"- **Which way the errors go.** Payments the company did not make / refusals where it paid, held-out: " +
        "; ".join(f"{NAME[k]} {_errors(r[k])}" for k in ("v1", "v2", "guarded", "tfidf_lr", "issuer_rate"))
        + f". No money figure is given: the records hold {', '.join(ph['record_fields'])} and no amount, and the "
        "dollar figures in the narratives are the consumer's own, unverified and not the refunded amount.",
        f"- **The issuer.** The issuer lookup's held-out kappa {R.ci(r['issuer_rate']['kappa'])} is the highest point "
        f"estimate, but TF-IDF cannot be told apart from it (kappa difference {_sg(h['tfidf_minus_issuer_kappa'])}, "
        f"AUC difference {_sg(df['issuer_rate_minus_tfidf_lr']['auc'])}), and its threshold (the TRAIN base rate) was "
        f"chosen after DEV; the planned company rule (raw TRAIN rates, the overall rate for issuers with few "
        f"complaints, threshold one half) gets {R.num(r['company_rule']['kappa']['value'])}, and the lookup's smoothed "
        f"rates at one half would give {R.num(smooth['kappa'])}. Masking issuer names lowers TF-IDF's AUC "
        f"({_sg(df['tfidf_masked_minus_tfidf_lr']['auc'])}). Within one issuer the text systems reach AUC "
        f"{R.num(min(x['value'] for x in wi.values()), 2)} to {R.num(max(x['value'] for x in wi.values()), 2)}, over "
        f"{wb['pairs']:,} same-issuer pairs from only {wb['issuers_with_pairs']} issuers. Resampling complaints, "
        f"{'no' if all(x['lo'] <= 0.5 for x in wi.values()) else 'not every'} lower bound clears 0.5: the best, "
        f"{NAME[top]}'s, is {R.num(wi[top]['lo'])} with {ph['auc_bootstrap_b']:,} resamples" +
        (f" and {R.num(w10['lo'])} with {w10['B']:,}" if top == "v2" else "") +
        f". Resampling the {wb['issuers_with_pairs']} issuers instead ({wb['B']:,} resamples), {len(above)} of the "
        f"{len(wb['systems'])} lower bounds lie above 0.5 (" + "; ".join(
            f"{NAME[k]} [{R.num(x['lo'])}, {R.num(x['hi'])}]" for k, x in wb["systems"].items()) +
        "). With so few issuers neither scheme is reliable, so whether the text beats a coin toss within an issuer is "
        f"not settled. On DEV, TF-IDF's within-issuer AUC was {R.ci(dv)}, resampling complaints. The lookup encodes "
        "each issuer's past habit: it shows what the label measures and is not a policy to deploy; inside one company "
        "it is a constant.",
        f"- **What the LLM adds over a lookup or TF-IDF** (restates the frozen claims): nothing this sample can show. "
        f"In {len(h['bonferroni_judge_vs_baseline'])} paired comparisons the judges are below in "
        f"{sum(1 for c in t['claims'] if c['id'] == 'judge_vs_baseline' and _claim_status(c) == 'decided')} and "
        "not distinguishable in the rest. This is a negative result for these two prompts on one local model, not for "
        "LLMs on this task.",
        f"- **The automation slice.** TF-IDF at its frozen threshold automated "
        f"{at['n_auto']}/{at['n']} at {R.pct(at['agreement']['value'])} ({_wilson(at)}; frozen claim: not shown to meet "
        f"the target). On those {sl['n_auto']} the issuer lookup would also have been right in "
        f"{sc['issuer_lookup_correct']} (TF-IDF is right in {sc['tfidf_correct']}), and TF-IDF with issuer names masked "
        f"makes the same decision in {sl['masked_same_decision']} of the {sl['n_auto']}; {sl['narrative_names_issuer']} "
        f"of the {sl['n_auto']} narratives name an issuer. The proposed refusals come from issuers that paid "
        f"{R.pct(sl['issuer_rate_auto_denied'], 0)} of their TRAIN complaints on average, the proposed refunds from "
        f"issuers that paid {R.pct(sl['issuer_rate_auto_refunded'], 0)}. The DEV operating point qualified on the boundary " +
        (f"({op['dev_automated']} cases, the minimum, at {R.pct(op['dev_agreement_on_automated'])}, the target) " if op else "") +
        f"and {sl['within_002_of_threshold']} held-out cases lie within 0.02 of the threshold. Whether a slice like this "
        f"exists inside a single company, where the issuer is fixed, this sample cannot show. Routing as measured: decide alone, none; for a person to approve, {at['auto_no_relief']} "
        f"proposed refusals (the company paid {at['wrong_no_relief']}) and {at['auto_relief']} proposed refunds (it paid "
        f"{at['auto_relief'] - at['wrong_relief']}); escalate, {at['n'] - at['n_auto']}. Whether pre-filled proposals "
        "save a reviewer time was not measured.",
        f"- **Gaming the judges** (a fixed set of {len(g['attacks'])} appended English sentences, one template each, on "
        f"{g['n_items']} DEV complaints, plus two controls: a neutral sentence, and the unchanged narrative sent again; "
        "a smoke test, not a red team). The fake internal note turned "
        f"{v2n['to_relief']} of {v2n['n_base_no_relief']} of judge v2's refusals and {v1n['to_relief']} of "
        f"{v1n['n_base_no_relief']} of judge v1's into refunds (the neutral sentence: {v1c['to_relief']} of "
        f"{v1c['n_base_no_relief']} for v1); with {v1n['n_base_no_relief']} refusals the test cannot tell v1 from the "
        f"control: it shows that v2 is moved and cannot show whether v1 is. Of the {hl['tests']} rewrite-against-control tests for the "
        f"two judges, {len(hl['survive'])} survives a Holm correction "
        f"({', '.join(x.replace('fake_case_note', 'fake note') for x in hl['survive'])}); these tests are two-sided and "
        "count flips in either direction. Counting only refusals turned into refunds, the direction an attacker wants, "
        f"with a one-sided exact test, {len(h1['survive'])} survive "
        f"({', '.join(x.replace('fake_case_note', 'fake note') for x in h1['survive'])}). "
        f"The input filter is seven English patterns: it caught the one wording it was written against "
        f"({f['attacked_flagged']['injection']}/{g['n_items']} copies), {f['paraphrases_flagged']} of "
        f"{f['paraphrases']} other wordings of the same instruction, {fr['further_flagged']} of {fr['further_wordings']} "
        "further wordings (obfuscated or reworded) written after the AI-agent reviews listed in `docs/deviations.md`, "
        f"{f['attacked_flagged']['fake_case_note']} of {g['n_items']} fake notes, and {real['flagged']} of "
        f"{real['n']:,} real narratives; it flags {fr['benign_flagged']} of {fr['benign']} harmless sentences that were "
        f"written to hit its patterns, so its flag is a routing lever the complainant holds. The guarded pipeline flipped {sv['guarded']['fake_case_note']['flips']}/"
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
        "refunds into refusals. The sentences were written with the published feature list in view. The keyword rule "
        "was not attacked.",
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
        f"{R.num(rt['v2']['shift']['max'], 2)}; judge v1 was about as repeatable on both "
        f"({rt['v1']['byte_identical']}/{rt['v1']['n']} and {nz['v1']['byte_identical']}/{nz['v1']['n']} byte-identical). "
        "Generation speed as logged during each pass (`results/run_log.jsonl`): " + "; ".join(
            f"{w} {' and '.join(str(x) for x in sp[(lab, v)])} tokens/s" for w, lab, v in (("judge v2 DEV retest", "replay", "v2"),
                                                          ("judge v2 held-out noise pass", "test noise pass", "v2"),
                                                          ("judge v1 DEV retest", "replay", "v1"),
                                                          ("judge v1 held-out noise pass", "test noise pass", "v1"))
            if (lab, v) in sp) +
        ". Memory was not recorded during either pass, so the cause of the DEV differences is not known. The "
        "confidence score is not stable enough to set a threshold on.",
        f"- **Cost** (compute on one laptop, an Apple M4 Max with 36 GB of memory, which the run logs do not record, and "
        f"a local model; DEV timings; no money cost or handle time was measured): a table lookup for the issuer rows; TF-IDF "
        f"{c['tfidf_lr']['predict_ms_per_item_one_by_one']:.1f} s of CPU per 1,000; judge v1 {c['v1']['hours_per_1000']:.1f} h, "
        f"judge v2 {c['v2']['hours_per_1000']:.1f} h" +
        (" (on the held-out run " + " and ".join(f"{v} {x:.1f} h" for v, x in held_h.items()) + ")" if held_h else "") +
        "; "
        "a person: " + (f"about {_people(m)[0][1]['median_seconds'] * 1000 / 3600:.0f} h, the median time per label "
                        f"({_people(m)[0][1]['median_seconds']:.0f} s) times 1,000, which leaves out breaks (the recorded "
                        "times include pauses with the page open); not a measured handle time."
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
    ex, wb, sc = ph["accuracy_vs_always_no_exact"], ph["within_issuer_by_issuer"], ph["tfidf_slice_correct"]
    out = [f"Computed by `src/judge/posthoc.py` after the held-out run, from the same data and caches; nothing here "
           f"was registered and none of it decides a claim. Kappa and accuracy intervals: {m['settings']['bootstrap_b']:,} "
           f"percentile-bootstrap resamples over complaints (paired where two systems are compared; DEV and held-out "
           f"resampled independently when they are compared); AUC intervals elsewhere in this report use "
           f"{ph['auc_bootstrap_b']:,} resamples, as the frozen code does.\n",
           "Held-out, by system: agreement with the record against always answering no relief (paired), the "
           "complaints on which exactly one of the two matches the record with the exact two-sided McNemar test on "
           f"them, the paired interval at the Bonferroni level for {len(ex['systems'])} comparisons "
           f"({R.pct(ex['level'], 3)}, same resamples), and the direction of the errors:\n",
           R.table(("System", "Matches the record", "Minus always no relief [95% CI]",
                    "Differs from always no: system right / always no right", "Exact p",
                    f"Minus always no relief, {R.pct(ex['level'], 3)} interval", "Would pay",
                    "Pays where the company did not", "Refuses where the company paid"),
                   [(SHORT[k], R.pct(r[k]["accuracy"]["value"]), _pts(x),
                     f"{ex['systems'][k]['system_only_right']} / {ex['systems'][k]['always_no_only_right']}",
                     ("below 0.001" if ex["systems"][k]["p_exact_two_sided"] < 0.001 else
                      f"{ex['systems'][k]['p_exact_two_sided']:.3f}"),
                     f"[{100 * ex['systems'][k]['lo']:+.1f}, {100 * ex['systems'][k]['hi']:+.1f}]",
                     _pays(r[k]), r[k]["confusion"]["none_relief"],
                     r[k]["confusion"]["relief_none"]) for k, x in h["accuracy_minus_always_no"].items()]),
           f"\nWhich system is highest was seen only after the run, so the highest row is the best of "
           f"{len(ex['systems'])} picked after the fact.",
           f"\nThe records hold {', '.join(ph['record_fields'])}; there is no amount field, so no error is given in money.",
           f"\nTF-IDF minus the issuer relief rate, held-out kappa (paired): {_sg(h['tfidf_minus_issuer_kappa'])}. "
           f"The issuer lookup's smoothed rates with the planned threshold of one half instead of the TRAIN base rate: "
           f"held-out kappa {R.num(ph['issuer_smoothed_at_half']['kappa'])} (the planned company rule, raw TRAIN rates "
           f"at one half: {R.num(r['company_rule']['kappa']['value'])}).",
           "\nThe six frozen judge-against-baseline differences at the Bonferroni level for six comparisons "
           f"({R.pct(next(iter(h['bonferroni_judge_vs_baseline'].values()))['level'], 2)} intervals, same resamples): " +
           "; ".join(f"{k.replace('_minus_', ' minus ')} {_sg(x)}" for k, x in h["bonferroni_judge_vs_baseline"].items()) + ".",
           "\nDEV against held-out kappa (independent samples; two-sided bootstrap p): " + "; ".join(
               f"{k.replace('_minus_v1_change', ' minus v1, change')} {_sg(x)} (p {x['p_two_sided']:.2f})"
               for k, x in ph["dev_vs_heldout"].items()) + ".",
           f"\nJudge v2's within-issuer AUC on held-out with {h['v2_within_issuer_at_10000']['B']:,} resamples: "
           f"[{R.num(h['v2_within_issuer_at_10000']['lo'])}, {R.num(h['v2_within_issuer_at_10000']['hi'])}], "
           f"{R.pct(h['v2_within_issuer_at_10000']['share_at_or_below_half'])} of resamples at or below 0.5. The "
           "within-issuer intervals elsewhere in this report resample complaints. Only "
           f"{wb['issuers_with_pairs']} held-out issuers carry a pair of a relief and a no-relief complaint "
           f"({wb['pairs']:,} pairs); resampling those issuers instead ({wb['B']:,} resamples):\n",
           R.table(("System", "Within-issuer AUC", "95% interval, issuers resampled", "Resamples at or below 0.5",
                    "95% interval, complaints resampled"),
                   [(SHORT[k], R.num(x["value"]), f"[{R.num(x['lo'])}, {R.num(x['hi'])}]",
                     R.pct(x["share_at_or_below_half"]), f"[{R.num(t['ranking']['systems'][k]['within_issuer_auc']['lo'])}, "
                     f"{R.num(t['ranking']['systems'][k]['within_issuer_auc']['hi'])}]")
                    for k, x in wb["systems"].items()]),
           f"\nWith {wb['issuers_with_pairs']} clusters neither scheme is reliable" +
           ("; the two disagree on whether the lower bounds clear 0.5." if
            any(x["lo"] > 0.5 for x in wb["systems"].values()) !=
            any(t["ranking"]["systems"][k]["within_issuer_auc"]["lo"] > 0.5 for k in wb["systems"]) else "."),
           f"\nThe TF-IDF slice automated at the frozen threshold ({h['tfidf_slice']['n_auto']} cases from "
           f"{h['tfidf_slice']['issuers']} issuers, the largest contributing {R.pct(h['tfidf_slice']['largest_issuer_share'], 0)}): "
           f"TF-IDF right in {sc['tfidf_correct']}, the issuer lookup right in {sc['issuer_lookup_correct']}; "
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
    rn = ph["filter_on_real_narratives"]
    out.append(f"\nInput filter on further wordings (written after the AI-agent reviews listed in `docs/deviations.md`, "
               f"checked against the filter only): {fr['further_flagged']} of {fr['further_wordings']} obfuscated or "
               f"reworded instructions flagged, {fr['benign_flagged']} of {fr['benign']} harmless sentences flagged "
               f"(written to hit the patterns), {fr['heldout_flagged']} of {fr['heldout_n']} held-out narratives and "
               f"{rn['flagged']} of {rn['n']:,} real narratives of TRAIN, DEV and held-out flagged (`FURTHER_WORDINGS` "
               "and `BENIGN` in `src/judge/posthoc.py`).")
    out.append(f"\nWhat the fake internal note changed in judge v2's extracted facts on the {gm['n']} gaming complaints: "
               "issuer status " + ", ".join(f"{k} {v}" for k, v in gm["status_changes"].items()) + "; issuer error " +
               ", ".join(f"{k} {v}" for k, v in gm["error_changes"].items()) + ". Fact-model weights (DEV fit): " +
               ", ".join(f"{k} {v:+.2f}" for k, v in gm["weights"].items()) + ".")
    small = ", ".join(f"{x['system']} {x['rewrite']} " + ("below 0.001" if x["p"] < 0.001 else f"{x['p']:.4f}")
                      for x in hl["smallest"])
    out.append(f"\nHolm correction over the {hl['tests']} rewrite-against-control tests of the two judges (exact "
               f"McNemar, two-sided, flips in either direction): surviving {', '.join(hl['survive']) or 'none'}; the "
               f"smallest p-values {small}.")
    h1 = ph["holm_one_sided"]
    out.append(f"\nThe same {h1['tests']} tests counting only refusals turned into refunds (exact McNemar, one-sided): "
               f"surviving Holm {', '.join(h1['survive']) or 'none'}; " + "; ".join(
                   f"{x['system']} {x['rewrite']} {x['only_rewrite']} / {x['only_control']}, p " +
                   ("below 0.001" if x["p_one_sided"] < 0.001 else f"{x['p_one_sided']:.4f}") for x in h1["rows"]) +
               " (complaints moved by the rewrite only / by the neutral control only).")
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
        f"Gist: {gi['url']} (unlisted: GitHub calls it secret, but anyone with the link can read it). Revisions, with "
        "the commit date GitHub's API reports for each:\n",
        R.table(("Revision: git commit date, as the API reports it", "Frozen commit it names",
                 "SHA-256 of results/frozen.json"), rows),
        f"\nA gist is a git repository, so these are git commit dates, which the client that writes a revision can "
        f"set. The gist's `created_at` ({gi['created_at'].replace('T', ' ').replace('Z', ' UTC')}) and `updated_at` "
        f"({gi['updated_at'].replace('T', ' ').replace('Z', ' UTC')}) are fields GitHub's server maintains; they " +
        ("agree with the first and the last revision." if (gi["created_at"], gi["updated_at"]) ==
         (gi["revisions"][0]["committed_at"], gi["revisions"][-1]["committed_at"]) else
         "do not agree with the first and the last revision."),
        f"\nThe run marker (`results/test_run.json`) records the start of the held-out run at "
        f"{t.get('started_at', 'n/a').replace('T', ' ').replace('+00:00', ' UTC')}, from commit "
        f"`{t.get('commit', '')[:12]}`; the frozen file now on disk has SHA-256 `{(m.get('frozen') or {}).get('sha256', '')[:16]}...`. "
        + ("Both match the gist's last revision (`make check` compares them with the hashes copied from it). " if ok else
           "They do not match the gist's last revision. ") +
        f"The method was frozen in {len(FREEZES)} commits ({', '.join(f'`{c}`' for c in FREEZES)}), each replacing "
        f"the one before (the first was removed again before the second was made); the gist records the last two, "
        "and the request journal `results/cache/sent.jsonl` " +
        (f"begins at the run marker ({ph['journal_first_at'].replace('T', ' ').replace('+00:00', ' UTC')})."
         if ph["journal_first_at"] == t.get("started_at") else
         f"begins at {ph['journal_first_at']}, not at the run marker."),
        f"\nWhat this shows: by the time of the last revision, {gap} before the run marker, the holder of the "
        "author's GitHub account had the frozen commit hash and the hash of the frozen file, and a commit hash fixes "
        "every file of that commit. What it does not show: that no held-out complaint had been sent to the model "
        "before then (that rests on the request journal and the caches in this repository), or anything about the "
        "times of other commits. A gist is controlled by its owner and is not a notarised record, and GitHub is not "
        "an independent timestamping service.",
        f"\nAnswer files at the run commit `{ph['integrity']['run_commit']}`, recorded after the run (they were not in "
        "the run marker): " + "; ".join(f"`{k}` {v['bytes']:,} bytes, SHA-256 `{v['sha256'][:16]}...`"
                                        for k, v in ph["integrity"]["run_files"].items()) +
        ". `make check` fails if these bytes change; they anchor later changes from the time that commit was first "
        "published, not from the time of the run."])


# ------------------------------------------------------------------ the blind labels
def _head(h) -> str:
    t = h["translations"]
    return (f"Sample: {h['sample_n']} {h['split'].upper()} complaints drawn by a hash of the ID "
            "(`data/label_sample.json`), which no model had answered and the annotator had not seen; labelled blind by "
            "the author against `docs/labelling_rubric.md`, which gives the person the rules the judge v1 prompt was "
            "given. Person against v1 therefore measures whether the model applies a written policy the way a person "
            "reading the same text does; person against v2 compares answers to different questions. The labels are one "
            "non-expert's reading of the English original, with a machine translation shown beside it, not ground "
            "truth. Blind means that the annotator does not see the recorded outcome, the company or any model answer "
            "for a case; he was not kept away from this repository, which publishes its hypotheses, the base rate and "
            "judge v2's cues. The analysis was registered before "
            f"any label existed (`docs/analysis_plan_human_labels.md`, {h['plan_registered_at'][:16].replace('T', ' ')} UTC, "
            f"amended {h['amended_at'][:16].replace('T', ' ')} UTC). The Russian reading aid beside the English original "
            "was made by a separate model, not the judged one (Anthropic's `claude-fable-5-2`, run as sub-agents of the "
            "AI coding assistant that wrote this repository, so those sub-agents read the 100 narratives), never part of "
            f"a judged input ({t['flagged']} translations fail a completeness check, {t['missing']} are missing).")


REGISTERED_VS_PERSON = ("v1", "v2", "tfidf_lr", "keyword_rule", "majority", "guarded")   # the plan's list
REGISTERED_PAIRED = ("v1_minus_tfidf_lr", "v2_minus_tfidf_lr")                          # judge minus TF-IDF


def _reg(registered: bool) -> str:
    return "registered" if registered else NOT_REGISTERED


TINY = 1e-9                 # float noise around zero (a constant answer has kappa 0 in every resample)


def _sign(v: float) -> int:
    return 0 if abs(v) < TINY else 1 if v > 0 else -1


def _side(x: dict) -> str:
    return "above zero" if x["lo"] > TINY else "below zero" if x["hi"] < -TINY else "spanning zero"


def _sign_check(figures) -> str:
    """The registered rule: if the two readings of 'cannot tell' disagree in sign or the interval changes side of
    zero, the report says so and draws no conclusion from that figure. figures: (name, main, sensitivity)."""
    bad = [name for name, a, b in figures if _sign(a["value"]) != _sign(b["value"]) or _side(a) != _side(b)]
    if not figures:
        return ""
    if bad:
        return ("Registered check of the two readings ('cannot tell' left out, and counted as no money back): they "
                "disagree in sign or in the side of zero of the interval for " + "; ".join(bad) +
                ". No conclusion is drawn from these figures.")
    return ("Registered check of the two readings ('cannot tell' left out, and counted as no money back): they agree "
            "in sign and in the side of zero of the interval for every kappa and paired difference above.")


def _person_row(f) -> str:
    pv, sv = f["person_vs_recorded"], f["sensitivity"]["person_vs_recorded"]
    return (f"Person against the recorded company decision (registered): kappa {R.ci(pv['kappa'])}, accuracy "
            f"{R.ci(pv['accuracy'], R.pct)} on the {f['decided']} decided cases; with 'cannot tell' counted as no money "
            f"back, over all {f['n']}: kappa {R.ci(sv['kappa'])}, accuracy {R.ci(sv['accuracy'], R.pct)}.")


def _against_person(f) -> str:
    rows = []
    sens = f["sensitivity"]["systems_vs_person"]
    for n, x in f["systems_vs_person"].items():
        if n not in SHORT:
            continue
        c = x["confusion"]          # here the person's label is the reference
        note = {"v1": " (same question as the rubric)", "v2": " (a different question)"}.get(n, "")
        rows.append((f"{SHORT[n]}{note} ({_reg(n in REGISTERED_VS_PERSON)})", R.ci(x["kappa"]),
                     R.ci(x["accuracy"], R.pct), R.ci(sens[n]["kappa"]), R.ci(sens[n]["accuracy"], R.pct),
                     R.pct(x["predicted_positive_rate"]), f"{c['none_relief']} / {c['relief_none']}"))
    return R.table(("System against the person", "Kappa [95% CI], decided cases", "Accuracy [95% CI], decided cases",
                    "Kappa, 'cannot tell' as no money back", "Accuracy, 'cannot tell' as no money back",
                    f"System pays ({NOT_REGISTERED})",
                    f"System pays, person would not / person would pay, system would not ({NOT_REGISTERED})"), rows)


def _paired(f, ps) -> str:
    p = f["paired_vs_person"]
    if not p:
        return ""
    parts = []
    for k, x in p.items():
        a, b = k.split("_minus_")
        s = (ps or {}).get(k)
        parts.append(f"{SHORT[a]} minus {SHORT[b]} ({_reg(k in REGISTERED_PAIRED)}): kappa {_sg(x['kappa'])}, accuracy "
                     f"{_pts(x['accuracy'])}" +
                     (f"; with 'cannot tell' as no money back, kappa {_sg(s['kappa'])}, accuracy {_pts(s['accuracy'])}"
                      if s else ""))
    return "Paired differences against the person, decided cases (paired bootstrap): " + "; ".join(parts) + "."


def _figures(f, ps) -> list:
    out = [("person against the record, kappa", f["person_vs_recorded"]["kappa"],
            f["sensitivity"]["person_vs_recorded"]["kappa"])]
    out += [(f"{NAME[n]} against the person, kappa", x["kappa"], f["sensitivity"]["systems_vs_person"][n]["kappa"])
            for n, x in f["systems_vs_person"].items() if n in NAME]
    for k, x in f["paired_vs_person"].items():
        s = (ps or {}).get(k)
        if s:
            a, b = k.split("_minus_")
            out += [(f"{NAME[a]} minus {NAME[b]}, {w}", x[w], s[w]) for w in ("kappa", "accuracy")]
    return out


def _human_figures(f, ps) -> list[str]:
    out = ["\n" + _person_row(f), "\n" + _against_person(f)]
    if f["paired_vs_person"]:
        out.append("\n" + _paired(f, ps))
    out.append("\n" + _sign_check(_figures(f, ps)))
    return out


RULE_V3 = ("drafted from the calibration half only, frozen, run once on the evaluation half, and accepted only if the "
           "paired interval of kappa against the person, v3 minus the better of v1 and v2 on the calibration half, lies "
           "entirely above zero on the evaluation half")


def _v3(m) -> str:
    from .prompts import VERSIONS
    if "v3" in VERSIONS:
        return "A v3 prompt exists; its evaluation-half result is reported only through the registered rule."
    people = _people(m)
    d = (((m.get("posthoc") or {}).get("human") or {}).get(people[0][0], {}).get("v3_decision") if people else None)
    if not d:
        return (f"No v3 prompt exists. If one is written, it is {RULE_V3}, as registered; otherwise the report says "
                "that no improvement was demonstrated.")
    per = sorted(set(d["decided_per_half"].values()))
    hw = sorted(d["evaluation_paired_kappa_half_width"].values())
    return ("**No v3 prompt.** None was written, and none will be in this release: a decision taken after the labels "
            f"were locked, because each half holds {' or '.join(str(x) for x in per)} decided labels, too few for the "
            "registered acceptance rule to have a realistic chance of showing anything" +
            (f". On the evaluation half, the paired kappa intervals against the person that were computed reach "
             f"{R.num(hw[0], 2)} to {R.num(hw[-1], 2)} either side of their estimates, so v3 would have to beat the "
             f"comparator by roughly that much before its interval could clear zero ({AFTER_LOCK})" if hw else "") +
            f". The registered rule stays as written: a v3 prompt would be {RULE_V3}. That the registered v3 step was "
            "not carried out is recorded in `docs/deviations.md`.")


# ------------------------------------------------------------------ who did what (README)
def b_who(m):
    people = _people(m)
    rev = people[0][1]["error_review"] if people else None
    e = m["error_analysis"]
    tail = ("The AI coding assistant chose the data and the label, wrote or drafted the code, the prompts, the gaming "
            "rewrites, the defence and the first reading of the disagreements, wrote the rules that were frozen before "
            "the held-out run, and started that run" +
            (" and, after the lock, the run of the judges on the labelled sample" if people else "") +
            "; the reviews named in `docs/deviations.md` were separate AI-agent sessions, not people, and there is no "
            "independent second author. Where a code comment says \"by hand\" (for example the issuer alias list in "
            "`src/judge/baselines.py`), it means a fixed list written into the code by that assistant.")
    if not (people and rev and rev["reviewed"]):
        return ("**Who did what.** The author set the task and directed the work, ran it on his own laptop through an "
                "AI coding assistant, decided to publish the result as it came out, and is reading and labelling the "
                "complaints himself, blind. " + tail)
    f = people[0][1]["all"]
    return ("**Who did what.** The author set the task and directed the work, ran it on his own laptop through an AI "
            f"coding assistant, and decided to publish the result as it came out. He read and labelled the {f['n']} "
            "complaints of the labelling sample himself, blind; after the labels were locked he reviewed, on a second "
            f"screen, the {rev['reviewed']} individually sorted disagreements of judge v1 on DEV ({rev['confirmed']} "
            f"drafted readings confirmed, {len(rev['changed'])} changed). The drafted readings of the other "
            f"{e['n_disagreements'] - rev['reviewed']} of the {e['n_disagreements']} DEV disagreements are still the "
            "assistant's and unconfirmed. " + tail)


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
        ps = extra.get(who, {}).get("paired_sensitivity", {})
        out.append(f"\nAnnotator `{who}` (the author; not a lawyer; no second reader, so no inter-annotator agreement), "
                   f"locked {r['locked_at'][:16].replace('T', ' ')} UTC: {f['decided']} decided "
                   f"({R.pct(f['person_relief_share'])} of them money back; the companies paid in "
                   f"{R.pct(f['recorded_relief_share'])} of the sample), {f['cannot_tell']} cannot tell "
                   f"({R.pct(f['cannot_tell'] / f['n'], 0)}; registered). Time per label ({NOT_REGISTERED}): median "
                   f"{r['median_seconds']:.0f} s" +
                   (f", quartiles {s['p25']:.0f} s and {s['p75']:.0f} s, {s['under_30']} under 30 s" if s else "") +
                   ". The recorded times also run while the page is open between labels, so they include the "
                   "annotator's breaks" + (f" (the longest single record is {s['max'] / 60:.0f} minutes; {AFTER_LOCK})"
                                           if s and "max" in s else "") +
                   f"; their sum, {r['total_minutes']:.0f} minutes, is therefore not the time spent labelling.")
        if "person_vs_recorded" not in f:
            out.append("\nToo few decided cases for agreement figures.")
            continue
        out.append("\n" + person_table({"human": {"annotators": {who: r}}}))
        if not any(v in f["systems_vs_person"] for v in ("v1", "v2")):
            out.append("\nThe judges have not answered these complaints yet (`make judge-labelled`).")
        out += _human_figures(f, ps.get("all"))[1:]        # the person's row is in the table above
        out.append("\n" + _cannot_tell(f, r))
    out.append("\n" + _v3(m) + " The author's own reading of each sorted disagreement, from the second screen, is in "
               "the full report.")
    return "\n".join(out)


def _cannot_tell(f, r) -> str:
    """What the share of 'cannot tell' means for anyone planning to use labels like these. Facts only: the
    rubric's definition, the registered count and recorded relief rate, and whether reasons were recorded."""
    return (f"**'Cannot tell' in {f['cannot_tell']} of {f['n']}.** The rubric asks for 'cannot tell' when the text does "
            "not describe a transaction the reader can identify, or when the decision turns entirely on evidence the "
            "text does not give. For anyone planning to use labels like these: a reader of the complaint text alone "
            f"left {f['cannot_tell']} of {f['n']} complaints without a decision, so such labels cover only part of the "
            "cases, and which part is chosen by the reader, not at random (registered: the companies paid in "
            f"{R.pct(f['cannot_tell_recorded_relief'])} of the 'cannot tell' cases, {R.pct(f['recorded_relief_share'])} "
            "of the whole sample). Which of the two reasons applied to a case was not recorded as a field "
            f"({r['with_note']} of {f['n']} labels " + ("carries" if r["with_note"] == 1 else "carry") +
            " a free-text note), so the reasons are not counted.")


def b_human_detail(m):
    h = m["human"]
    if not h["annotators"]:
        return "_No locked set of labels yet._"
    extra = (m.get("posthoc") or {}).get("human", {})
    out = []
    for who, r in h["annotators"].items():
        f = r["all"]
        ps = extra.get(who, {}).get("paired_sensitivity", {})
        out.append(f"Annotator `{who}`. Cases answered 'cannot tell' (registered): {f['cannot_tell']}; recorded relief "
                   f"rate among them {R.pct(f['cannot_tell_recorded_relief'])}; relief rate of each system on them: " +
                   ", ".join(f"{SHORT.get(n, n)} {R.pct(v)}" for n, v in f["cannot_tell_system_relief"].items()) +
                   f". Notes written: {r['with_note']}. Cases shown with a flagged translation: "
                   f"{r['translation_flagged_when_shown']}" +
                   (f"; translation shown in {extra[who]['translation_shown']} of {f['n']} ({NOT_REGISTERED})."
                    if who in extra else "."))
        for half, g in r["halves"].items():
            out.append(f"\n{half.capitalize()} half ({g['n']} cases, {g['decided']} decided, {g['cannot_tell']} cannot "
                       "tell; registered); the registered plan reads these only through the v3 rule:")
            out += _human_figures(g, ps.get(half)) if "person_vs_recorded" in g else ["\nToo few decided cases."]
        e = r["error_review"]
        out.append(f"\nSecond screen (after the lock), the author's own reading of the {e['total']} individually sorted "
                   f"v1 DEV disagreements: {e['reviewed']} reviewed, {e['confirmed']} drafted readings confirmed, "
                   f"{len(e['changed'])} changed.")
        rows = (extra.get(who) or {}).get("review") or []
        if rows:
            out.append(f"\nCase by case ({NOT_REGISTERED}):\n")
            out.append(R.table(("Complaint (DEV)", "Drafted reading (AI coding assistant)", "Author's reading",
                                "Author's note"),
                               [(x["complaint_id"], x["drafted"].replace("_", " "),
                                 ("same" if x["confirmed"] else x["author"].replace("_", " ")), x["note"])
                                for x in rows]))
    return "\n".join(out)


# ------------------------------------------------------------------ data and licence (README)
def b_data_licence(m):
    ph = m.get("posthoc") or {}
    dw = ph.get("data_window")
    if not dw:
        return R.PENDING
    sp = dw["splits"]
    return (f"Filter: complaints received by the CFPB from {dw['filter_from']} up to, not including, "
            f"{dw['filter_before']}; the complaints in the records were received from {dw['first']} to {dw['last']}. "
            f"`data/records.jsonl.gz` holds {dw['records']:,} complaints (TRAIN {sp['train']:,}, DEV {sp['dev']}, held-out "
            f"{sp['test']}, and {sp['debug']} used only for smoke tests and timing), each with its narrative, product and "
            "issue fields, "
            f"company and recorded response. `results/cache/translations.jsonl` holds machine translations of "
            f"{dw['translations']['n']} narratives: the {dw['translations']['labelling_sample']} of the labelling sample "
            f"and the {dw['translations']['second_screen']} DEV disagreements of the second labelling screen; "
            "`results/error_analysis/` holds short quotes from DEV narratives.")
