"""The README's first screen, the held-out claims, the practitioner section, the exploratory section,
the freeze record and the human-label section. Written after the held-out run; presentation only
(results/post_run_changes.json records it). Numbers come from the frozen computation (analysis.py)
or, where marked exploratory, from posthoc.py.
"""
from __future__ import annotations

import re


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
# Seventh change (after the held-out run): where a block points the reader to a file or directory of the repository,
# the path is a relative Markdown link. The same block is rendered into the README (at the root) and into
# results/report.md (one level down), so report.render passes the path from the rendered file to the root under
# this key; the README gets no key and links from the root.
LINK_BASE = "_link_base"


def link(path: str, m) -> str:
    """`path` as a clickable relative link, with the path kept as code in the link text."""
    return f"[`{path}`]({m.get(LINK_BASE, '')}{path})"


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
        f"charge. Most came after the card issuer had already handled the dispute. Of the "
        f"{m['base_rates']['train']['n']:,} training complaints, {R.pct(share, 0)} are filed as 'company isn't "
        "resolving a dispute'. On "
        f"the {m['error_analysis']['n_dev']} development complaints, the second prompt (judge v2, below) reported that "
        f"the issuer had already refused in {pdn['n_denied']}; a cruder keyword search finds a refusal in "
        f"{R.pct(proxy['n_with'] / m['cues']['n_train'], 0)} of the training complaints. The label is how the "
        "company closed the complaint with the regulator, with money back (goodwill credits included) or without, so "
        "every figure here is agreement with what the company did, not correctness." + _person_first_screen(m),
        "",
        f"- **Always answering \"no money back\" matches the record in {R.pct(r['majority']['accuracy']['value'])} of "
        f"the {n} held-out complaints" +
        (", and no system is clearly above it.**" if not clearly else ".**") +
        f" {EXPL} The other {len(accs)} systems match the record in {R.pct(min(accs))} to {R.pct(max(accs))}. Agreement flatters systems "
        "that mostly say no, so the plan scores them by kappa (agreement beyond what the base rate gives), where the "
        + " and the ".join(f"{NAME[k]} ({R.num(r[k]['kappa']['value'])})" for k in lead) + " lead.",
        f"- **A model applying the written refund rules to the consumer's account (judge v1) would pay "
        f"{_pays(r['v1'])} of the {n}; the companies paid {r['majority']['confusion']['relief_none']}.** {EXPL} It errs "
        f"towards paying ({r['v1']['confusion']['none_relief']} payments the company did not make, "
        f"{r['v1']['confusion']['relief_none']} refusals where it paid)" +
        (f"; {' and '.join(('the ' if k == 'guarded' else '') + NAME[k] for k in toward_pay)} err towards refusing "
         "(" + "; ".join((f"{NAME[k]}: {r[k]['confusion']['none_relief']} payments the company did not make, "
                          f"{r[k]['confusion']['relief_none']} refusals where it paid") if j == 0 else
                         (f"{'the ' if k == 'guarded' else ''}{NAME[k]}: {r[k]['confusion']['none_relief']} and "
                          f"{r[k]['confusion']['relief_none']}") for j, k in enumerate(toward_pay)) + ")."
         if toward_pay else "."),
        f"- **The record follows which issuer answered** {EXPL}: knowing the bank does about as well as reading "
        "the complaint. A lookup of each issuer's past payout rate, which never reads the text, agrees with the record "
        f"about as well as the best text model (kappa {R.num(r['issuer_rate']['kappa']['value'])} against "
        f"{R.num(r[text_best]['kappa']['value'])}, difference {_sg(ph['heldout']['tfidf_minus_issuer_kappa'])}). Hiding "
        f"the issuer's name from the text model makes it rank complaints worse (AUC "
        f"{_sg(df['tfidf_masked_minus_tfidf_lr']['auc'])}). Comparing only complaints about the same issuer, the text "
        f"systems put paid complaints above unpaid ones only weakly (AUC {R.num(lo_w, 2)} to {R.num(hi_w, 2)}, where "
        "0.5 is a coin toss), and whether even that is better than chance is not settled: the answer changes with "
        f"what is resampled, complaints or issuers, and only {wb['issuers_with_pairs']} issuers have both a paid and "
        "an unpaid complaint to compare.",
        f"- **Decision: automate nothing.** The plan required {R.pct(m['settings']['target_agreement'], 0)} agreement "
        "on the cases a system would decide alone, and no system is shown to reach it. The only system with a frozen "
        f"threshold, a classic text classifier (TF-IDF), would decide {at['n_auto']} of the {at['n']} alone and agrees "
        f"on {R.pct(at['agreement']['value'])} of them, but on so few cases the interval runs from "
        f"{R.pct(at['agreement_wilson'][0])} to {R.pct(at['agreement_wilson'][1])} (Wilson). In a test on development "
        f"complaints a fake internal note appended to the end of the text turned {v2n['to_relief']} of "
        f"{v2n['n_base_no_relief']} of judge v2's refusals into refunds, and a sentence naming four banks, written "
        f"with TF-IDF's published feature list in view, turned {tg['to_relief']} of {tg['n_denials']} of TF-IDF's refusals "
        "on the same complaints into refunds.",
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
    out.append("- Two frozen items are records of the procedure, not tests, so they could not have come out "
               "otherwise; they are listed only to complete the list: " +
               (f"**{llm}**, so none of them had a threshold to apply on held-out; " if llm else "") +
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


FAIR = 0.4                  # the upper end of what is conventionally called fair agreement (kappa)


def _includes_zero(x: dict) -> bool:
    return x["lo"] <= TINY and x["hi"] >= -TINY


def _same_range(f, ex):
    """The range of the eight systems' kappa against the record on the person's decided cases, and whether the
    person's kappa lies within it (descriptive; no difference was tested)."""
    same = (ex.get("kappa_ranges_vs_record") or {}).get("same_decided_cases")
    if not same:
        return None, None
    k = f["person_vs_recorded"]["kappa"]["value"]
    return same, ("within" if same["min"] <= k <= same["max"] else "outside")


def _person_first_screen(m) -> str:
    if not _people(m):
        return ""
    _, f, ex = _person_facts(m)
    pv = f["person_vs_recorded"]["kappa"]
    same, where = _same_range(f, ex)
    t = (m.get("test") or {}).get("results") or {}
    text_max = max((t[k]["kappa"]["value"] for k in TEXT_SYSTEMS if k in t), default=None)
    weak = _includes_zero(pv) and text_max is not None and text_max < FAIR
    return (f" A person reading only the complaint, blind (one annotator, the author), could not decide "
            f"{f['cannot_tell']} of {f['n']}; on the {f['decided']} he decided, his agreement with the record is kappa "
            f"{R.ci(pv)} (registered)" + (", an interval that includes zero" if _includes_zero(pv) else "") +
            (f", {where} the spread of the systems on the same cases ({R.num(same['min'])} to {R.num(same['max'])}; "
             f"{AFTER_LOCK}; no difference tested)" if same else "") + "." +
            (" For this annotator, as for the text systems, the complaint text alone was a weak guide to what the "
             "company did (see below)." if weak else " See below."))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _judge_sides(sp: dict, sides: dict, decided: int) -> str:
    """Which side of zero each judge's interval against the person lies on, and, where a bound sits at zero and
    moves to either side with the bootstrap seed (posthoc.seed_sides), that this is not settled."""
    groups, out = {}, []
    for v in ("v1", "v2"):
        if v not in sp:
            continue
        x, s = sp[v]["kappa"], (sides or {}).get(f"{v}_vs_person:kappa")
        if s and len(s["main"]["sides"]) > 1:
            lo = s["main"]["lo_min"] <= TINY < s["main"]["lo_max"]
            a, b = (s["main"]["lo_min"], s["main"]["lo_max"]) if lo else (s["main"]["hi_min"], s["main"]["hi_max"])
            out.append(f"{_cap(NAME[v])}'s {'lower' if lo else 'upper'} bound sits at zero: from {a:+.3f} to {b:+.3f} "
                       f"over {len(SEEDS_OF(sides))} bootstrap seeds ({AFTER_LOCK}), so whether its interval clears zero "
                       f"depends on the bootstrap seed, and nothing is concluded for {NAME[v]} either way.")
        else:
            side = ("includes zero" if _includes_zero(x) else "lies entirely above zero" if x["lo"] > TINY else
                    "lies entirely below zero")
            groups.setdefault(side, []).append(NAME[v])
    head = [(f"{_cap(' and '.join(n))}'s interval {side}." if len(n) == 1 else
             f"The intervals of {' and '.join(n)} {side.replace('includes', 'include').replace('lies', 'lie')}.")
            for side, n in groups.items()]
    return " ".join(head + out)


def SEEDS_OF(sides: dict) -> list:
    return sides.get("_seeds") or []


def _sides(ex: dict, scope: str) -> dict:
    """The seed sides of one scope, with the list of seeds under '_seeds'."""
    sd = ex.get("seed_sides") or {}
    d = dict((sd.get("scopes") or {}).get(scope) or {})
    if d:
        d["_seeds"] = sd.get("seeds") or []
    return d


def human_summary(m) -> str:
    """The registered human-label figures in a few lines for the held-out section; the full tables print
    once, in the human-label section."""
    who, f, ex = _person_facts(m)
    row = m["human"]["annotators"][who]
    pv, sv = f["person_vs_recorded"]["kappa"], f["sensitivity"]["person_vs_recorded"]["kappa"]
    sp = f["systems_vs_person"]
    same, _ = _same_range(f, ex)
    hv = {k: g for k, g in row["halves"].items() if "person_vs_recorded" in g}
    dec = sorted({g["decided"] for g in hv.values()})
    halves = ("; " + " and ".join(f"{R.num(g['person_vs_recorded']['kappa']['value'])} on the {k} half"
                                  for k, g in hv.items()) +
              f" ({dec[0]} decided each" if len(dec) == 1 else
              f" ({' and '.join(str(g['decided']) for g in hv.values())} decided") + "; registered)" if hv else ""
    where = (f" The {same['n']} systems' kappas against the record on the same {f['decided']} cases run from "
             f"{R.num(same['min'])} ({NAME[same['min_system']]}) to {R.num(same['max'])} ({NAME[same['max_system']]}; "
             f"{AFTER_LOCK}); no difference between the person and any system was tested." if same else "")
    ct = ex.get("cannot_tell_counted")
    counted = (f"; counted as money back, {R.num(ct['as_money_back'])}, and from {R.num(ct['as_the_opposite'])} to "
               f"{R.num(ct['as_the_record'])} if every 'cannot tell' took the opposite of the record or the record "
               f"itself ({AFTER_LOCK})" if ct else "")
    judges = "; ".join(f"{NAME[v]} {R.ci(sp[v]['kappa'])}" for v in ("v1", "v2") if v in sp)
    sides = _sides(ex, "all")
    return "\n".join([
        f"**Blind human labels** (a separate analysis on {f['n']} TRAIN complaints, not the held-out split; "
        "registered in `docs/analysis_plan_human_labels.md` before any label existed; full tables under \"A person's "
        "blind labels\" below):\n",
        f"- One annotator, the author: not a lawyer, and not a native English reader, who read the English original "
        f"with a machine translation beside it. He labelled {f['n']} complaints blind and could not decide "
        f"{f['cannot_tell']} of them from the text (registered).",
        f"- On the {f['decided']} he decided, his agreement with the company's record is kappa {R.ci(pv)} (registered)" +
        (", an interval that includes zero" if _includes_zero(pv) else "") + halves + "." + where +
        f" With 'cannot tell' counted as no money back, over all {f['n']} (registered sensitivity line): kappa "
        f"{R.ci(sv)}{counted}.",
        f"- The judges against the person, decided cases (registered): {judges}. " +
        _judge_sides(sp, sides, f["decided"]) + " " + _check(f, ex.get("paired_sensitivity", {}).get("all"), sides),
    ])


# ------------------------------------------------------------------ practitioner reading
def _person_cost(m) -> str:
    """The person's time per 1,000 labels from the median, and from the mean with long records capped (posthoc)."""
    who, r = _people(m)[0]
    med = r["median_seconds"]
    s = ((((m.get("posthoc") or {}).get("human") or {}).get(who) or {}).get("seconds") or {})
    f = r["all"]
    capped = (f", or about {s['mean_capped'] * 1000 / 3600:.0f} h from the mean time per label with each record capped "
              f"at {s['cap_s'] // 60:.0f} minutes, so that a record that includes a long break counts for "
              f"{s['cap_s'] // 60:.0f} minutes ({s['mean_capped']:.0f} s; {AFTER_LOCK})" if "mean_capped" in s else "")
    return (f"about {med * 1000 / 3600:.0f} h at the median time per label ({med:.0f} s) times 1,000 labels (at this "
            f"annotator's rate about {round(1000 * f['decided'] / f['n']):,} of them would be decisions){capped}; the "
            "recorded times include pauses with the page open, so neither is a measured handle time.")


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
    n_t = t["n"]
    route = (f"Routing as measured, at TF-IDF's frozen threshold: decide alone, none; for a person to approve, "
             f"{at['auto_no_relief']} proposed refusals (the company paid {at['wrong_no_relief']}) and {at['auto_relief']} "
             f"proposed refunds (it paid {at['auto_relief'] - at['wrong_relief']}); escalate, {at['n'] - at['n_auto']}. "
             "Whether pre-filled proposals save a reviewer time was not measured. ")
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
        f"- **If one of these decided alone, on these {n_t} held-out complaints** (the company paid "
        f"{r['majority']['confusion']['relief_none']}): " + "; ".join(
            (f"{NAME[k]} would pay {r[k]['confusion']['none_relief']} that the company did not pay and refuse "
             f"{r[k]['confusion']['relief_none']} that it paid" if j == 0 else
             f"{'the ' if k in ('guarded', 'tfidf_lr', 'issuer_rate') else ''}{NAME[k]} would pay "
             f"{r[k]['confusion']['none_relief']} and refuse {r[k]['confusion']['relief_none']}")
            for j, k in enumerate(("v1", "v2", "guarded", "tfidf_lr", "issuer_rate"))) +
        ". 'Paid' and 'did not pay' are the company's record, not a correct decision"
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
        f"- **The automation slice.** {route}TF-IDF at its frozen threshold automated "
        f"{at['n_auto']}/{at['n']} at {R.pct(at['agreement']['value'])} ({_wilson(at)}; frozen claim: not shown to meet "
        f"the target). On those {sl['n_auto']} the issuer lookup would also have been right in "
        f"{sc['issuer_lookup_correct']} (TF-IDF is right in {sc['tfidf_correct']}), and TF-IDF with issuer names masked "
        f"makes the same decision in {sl['masked_same_decision']} of the {sl['n_auto']}; {sl['narrative_names_issuer']} "
        f"of the {sl['n_auto']} narratives name an issuer. The proposed refusals come from issuers that paid "
        f"{R.pct(sl['issuer_rate_auto_denied'], 0)} of their TRAIN complaints on average, the proposed refunds from "
        f"issuers that paid {R.pct(sl['issuer_rate_auto_refunded'], 0)}. The DEV operating point qualified on the boundary " +
        (f"({op['dev_automated']} cases, the minimum, at {R.pct(op['dev_agreement_on_automated'])}, the target) " if op else "") +
        f"and {sl['within_002_of_threshold']} held-out cases lie within 0.02 of the threshold. Whether a slice like this "
        "exists inside a single company, where the issuer is fixed, this sample cannot show.",
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
        "further wordings (obfuscated or reworded) written after the AI-agent reviews listed in "
        f"{link('docs/deviations.md', m)}, "
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
        "a person: " + (_person_cost(m) if _people(m) else "not measured until the blind labels are locked."),
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
        "whether a reviewer can use the output. **Before trying again**: a label that says what should have happened "
        "under a written policy, set by trained agents, not how a company closed a regulator complaint; the system of "
        "record beside the claimant's text (the transaction, the merchant's response, the amount, the scheme "
        "deadlines); handle time and money measured for the person and for the system on the same cases; an attack "
        "set written by someone other than the defence author and held out from tuning, with the claimant's text read "
        "as evidence and never as an instruction; and enough cases in the automated slice that the lower end of its "
        "interval, not the estimate, clears the target.",
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
    out.append("\nInput filter on further wordings (written after the AI-agent reviews listed in "
               f"{link('docs/deviations.md', m)}, "
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
            "truth. Blind means that the annotator does not see the recorded outcome, the company field or any model answer "
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


def _sign_check(figures, unsettled=(), registered=None) -> str:
    """The registered rule: if the two readings of 'cannot tell' disagree in sign or the interval changes side of
    zero, the report says so and draws no conclusion from that figure. figures: (name, main, sensitivity).
    unsettled: names whose side of zero moves with the bootstrap seed (posthoc.seed_sides); they are left out of the
    check and named, because the check would come out either way. registered: the names the plan registers (None:
    all); the others are reported apart, as descriptive."""
    if not figures:
        return ""
    head = "Registered check of the two readings ('cannot tell' left out, and counted as no money back): "
    bad = [name for name, a, b in figures if name not in unsettled
           and (_sign(a["value"]) != _sign(b["value"]) or _side(a) != _side(b))]
    reg = [n for n in bad if registered is None or n in registered]
    desc = [n for n in bad if registered is not None and n not in registered]
    out = [head + ("they disagree in sign or in the side of zero of the interval for " + "; ".join(reg) + "."
                   if reg else "they agree in sign and in the side of zero of the interval for every " +
                   ("registered " if registered is not None else "") + "kappa and paired difference above" +
                   (" that the check can be applied to." if unsettled else "."))]
    if desc:
        out.append(f"The same reading of figures {NOT_REGISTERED.replace('; descriptive', '')} (descriptive): they "
                   "disagree for " + "; ".join(desc) + ".")
    uns = [name for name, _, _ in figures if name in unsettled]
    if uns:
        out.append("For " + "; ".join(uns) + ", a bound of the interval sits at zero and moves to either side of it "
                   "with the bootstrap seed, so the check cannot be applied" +
                   (" to it." if len(uns) == 1 else " to them."))
    if bad or uns:
        out.append("No conclusion is drawn from these figures.")
    return " ".join(out)


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


def _figure_rows(f, ps) -> list:
    """(key in posthoc.seed_sides, name, main reading, sensitivity reading, registered) of every figure the
    registered check reads."""
    out = [("person_vs_recorded:kappa", "person against the record, kappa", f["person_vs_recorded"]["kappa"],
            f["sensitivity"]["person_vs_recorded"]["kappa"], True)]
    out += [(f"{n}_vs_person:kappa", f"{NAME[n]} against the person, kappa", x["kappa"],
             f["sensitivity"]["systems_vs_person"][n]["kappa"], n in REGISTERED_VS_PERSON)
            for n, x in f["systems_vs_person"].items() if n in NAME]
    for k, x in f["paired_vs_person"].items():
        s = (ps or {}).get(k)
        if s:
            a, b = k.split("_minus_")
            out += [(f"{k}:{w}", f"{NAME[a]} minus {NAME[b]}, {w}", x[w], s[w], k in REGISTERED_PAIRED)
                    for w in ("kappa", "accuracy")]
    return out


def _figures(f, ps) -> list:
    return [(n, a, b) for _, n, a, b, _ in _figure_rows(f, ps)]


def _check(f, ps, sides=None) -> str:
    rows = _figure_rows(f, ps)
    uns = {n for k, n, *_ in rows if k in (sides or {}) and not sides[k]["settled"]}
    return _sign_check([(n, a, b) for _, n, a, b, _ in rows], uns, {n for _, n, _, _, r in rows if r})


def _human_figures(f, ps, sides=None) -> list[str]:
    out = ["\n" + _person_row(f), "\n" + _against_person(f)]
    if f["paired_vs_person"]:
        out.append("\n" + _paired(f, ps))
    out.append("\n" + _check(f, ps, sides))
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
    return ("**No v3 prompt.** The registered v3 step was not carried out in this release, and the registered rule "
            f"stands: a v3 prompt would be {RULE_V3}. Not writing one was decided after the labels were locked and "
            "read. The plan had already said that at the size of a half only a large improvement could pass and that "
            "'no improvement demonstrated' was the likely outcome, and it registered the step all the same, so leaving "
            f"it out is a departure from the plan, recorded in {link('docs/deviations.md', m)}, not something the plan "
            "provides "
            f"for. For scale: each half holds {' or '.join(str(x) for x in per)} decided labels" +
            (f", and on the evaluation half the paired kappa intervals against the person that were computed reach "
             f"{R.num(hw[0], 2)} to {R.num(hw[-1], 2)} either side of their estimates ({AFTER_LOCK})" if hw else "") +
            ".")


# ------------------------------------------------------------------ who did what (README)
def _screen(m, who):
    """The timing of the second screen (posthoc), or None."""
    return (((m.get("posthoc") or {}).get("human") or {}).get(who) or {}).get("second_screen")


def _went_through(rev, sc) -> str:
    """'went through the 24 ... in about 15 minutes (median 32 s between answers)', from the record only."""
    n = "all " + str(rev["total"]) if rev["reviewed"] == rev["total"] else f"{rev['reviewed']} of the {rev['total']}"
    t = (f" in about {sc['minutes_first_to_last']:.0f} minutes from the first answer to the last (median "
         f"{sc['median_gap_s']:.0f} s between answers)" if sc and sc.get("median_gap_s") is not None else "")
    notes = ("" if not sc else " and left no notes" if not sc["notes"] else
             f" and wrote {sc['notes']} note" + ("" if sc["notes"] == 1 else "s"))
    return n, t, notes


def b_who(m):
    people = _people(m)
    rev = people[0][1]["error_review"] if people else None
    e = m["error_analysis"]
    tail = ("It wrote the rules that were frozen before the held-out run and started that run" +
            (" and, after the lock, the run of the judges on the labelled sample" if people else "") +
            f". The reviews named in {link('docs/deviations.md', m)} were separate AI-agent sessions, not people, and "
            "there is no "
            "independent second author. Where a code comment says \"by hand\" (for example the issuer alias list in "
            "`src/judge/baselines.py`), it means a fixed list written into the code by that assistant.")
    if not (people and rev and rev["reviewed"]):
        return ("**Who did what.** The author set the task and directed the work, run on his own laptop, through an AI coding "
                "assistant, and decided to publish the result as it came out. He is labelling the complaints of the "
                "labelling sample himself, blind. The AI coding assistant chose the data and the label and wrote or "
                f"drafted the code (after the held-out run also the SQL queries in {link('sql/', m)}), the prompts, "
                "the gaming "
                "rewrites, the defence and the first reading of the "
                f"{e['n_disagreements']} DEV disagreements. " + tail)
    who, f = people[0][0], people[0][1]["all"]
    n, t, notes = _went_through(rev, _screen(m, who))
    return ("**Who did what.** The author set the task and directed the work, run on his own laptop, through an AI coding "
            "assistant, and decided to publish the result as it came out. He chose to label all "
            f"{f['n']} complaints of the labelling sample himself rather than a smaller set, labelled them blind, and "
            f"after the lock went through {n} disagreements of judge v1 on DEV that the assistant had sorted one by one "
            f"(the other {e['n_category_only']} carry only a category) on a second screen{t}: "
            f"he confirmed {rev['confirmed']} drafted readings, changed {len(rev['changed'])}{notes}. The AI coding "
            "assistant chose the data and the label and wrote or drafted the code (after the held-out run also the SQL "
            f"queries in {link('sql/', m)}, which it chose and wrote when the author asked for an SQL section), the "
            "prompts, the "
            "gaming rewrites, "
            f"the defence and the first reading of the {e['n_disagreements']} DEV disagreements, "
            f"{e['n_disagreements'] - rev['reviewed']} of which no person has confirmed. " + tail)



def b_sql_queries(m):
    """A pointer to sql/ (sixth post-run change): which reported figures its queries recompute, and the descriptive
    breakdown they print for the first time (added later; descriptive; posthoc.issuer_volume and
    posthoc.within_issuer_by_issuer.pairs_by_issuer)."""
    if not _ready(m):
        return R.PENDING
    ir = m["test"]["results"]["issuer_rate"]
    wb = m["posthoc"]["within_issuer_by_issuer"]
    iv = m["posthoc"]["issuer_volume"]
    lo, hi = _within_issuer_range(m)
    top = sorted(wb["pairs_by_issuer"].items(), key=lambda kv: (-kv[1], kv[0]))[:2]
    return (f"{link('sql/', m)} holds eight short, commented SQL queries, run in SQLite through Python's built-in "
            "`sqlite3`. Most "
            "of what they print recomputes figures this study already reports (the README and `results/report.md`) "
            f"from the same data: the relief base rate per split ({R.pct(m['base_rates']['test']['base_rate'])} "
            "held-out); the issuer lookup's TRAIN payout rates (GROUP BY) and its decision for each held-out complaint "
            "(a LEFT JOIN on the issuer, with the rule for issuers unseen in TRAIN), with its agreement "
            f"({R.pct(ir['accuracy']['value'])}) and kappa ({R.num(ir['kappa']['value'])}) computed in SQL from the 2x2 "
            "counts; the agreement, 'pays', 'refuses' and kappa columns of the held-out table for every system in one "
            f"query; the {wb['pairs']:,} same-issuer pairs from {wb['issuers_with_pairs']} issuers, counted by a "
            f"self-join and by a product of grouped counts, and the within-issuer AUC over them ({R.num(lo, 2)} to "
            f"{R.num(hi, 2)} for the text systems); and the issuer facts and the dates of the full report, with window "
            "functions. They also print a breakdown that was not reported before (added later; descriptive): each "
            "issuer's TRAIN count, paid count, raw and smoothed rate, share of TRAIN and running share (the "
            f"{iv['k']} largest issuers hold {R.pct(iv['share_of_k_largest'])} of TRAIN), and the same-issuer pairs per "
            "issuer (" + " and ".join(f"{g} {k:,}" for g, k in top) + f" of the {wb['pairs']:,}). `make sql` prints "
            "the results; `tests/test_sql.py` checks every value of every result, row by row and in order, against the "
            "same figure computed in Python from the same data (by the frozen code where it computes one) and against "
            f"`results/metrics.json`, not against the queries' own output; {link('docs/sql.md', m)} explains every "
            "query clause by "
            "clause. The AI coding assistant chose and wrote the queries after the held-out run and the human labels, "
            "when the author asked for an SQL section. They decide nothing and feed no reported number, and the "
            "intervals, which come from resampling, are not reproduced.")


# ------------------------------------------------------------------ deviations, in short
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
TIMES = ("no times", "once", "twice", "three times", "four times", "five times", "six times", "seven times",
         "eight times", "nine times", "ten times")
# one line per post-run change, as recorded in docs/deviations.md; the count comes from the records
CHANGES_IN_SHORT = {
    1: "reorganising the README so that the held-out result comes first",
    2: "adding the exploratory analyses above",
    3: "correcting some of their wording after two further AI-agent reviews, adding the figures behind those "
       "corrections and a check of the Python version to the `Makefile`",
    4: "after the blind labels were locked, printing the human-label tables once, with a short summary in the "
       "held-out section, and stating the time per label, the reading record and the decision not to write a v3 "
       "prompt",
    5: "after final reviews, rewording sentences on the human labels that read stronger than the data allow, "
       "anchoring the locked labels under the check, and removing three unused personal-data columns from the label "
       "cross-check file",
    6: "after two further AI-agent reviews, adding the SQL queries in `sql/` that recompute reported figures and "
       "print a descriptive breakdown not reported before, and correcting wording published at commit `72b077a`: in "
       "the README the key to the tags, the first screen, 'Who did what', the human-label sections and the person's "
       "cost line, and in the full report the wording of the second screen and of the issuer rates",
    7: "removing, at the author's decision, a clause about his own level from the introduction of the SQL "
       "walk-through, without changing who is stated to have written the SQL, and making the README's pointers to "
       "the SQL, its walk-through and the deviations list clickable links",
    8: "registering a plan (`docs/analysis_plan_models.md`) to run the same two prompts on four other local models, "
       "on DEV and the labelled sample only, before any of them was sent a complaint",
}
_ORDINAL_CHANGE = re.compile(r"\b(" + "|".join(ORDINALS) + r")\s+change\b", re.I)


def post_run_change_count(m) -> int:
    """How many changes after the held-out run results/post_run_changes.json records: the highest 'Nth change' that
    the accepted records name (a record names every change that touched its file)."""
    recs = (m.get("frozen") or {}).get("post_run_changes") or []
    return max((ORDINALS.index(w.lower()) + 1 for r in recs for w in _ORDINAL_CHANGE.findall(r["what"])), default=0)


def b_deviations_in_short(m):
    """The deviations list in short, with the number of post-run changes counted from the records, so that the
    count written here cannot fall behind them."""
    n = post_run_change_count(m)
    if set(CHANGES_IN_SHORT) != set(range(1, n + 1)):
        raise RuntimeError(f"results/post_run_changes.json records {n} post-run changes; CHANGES_IN_SHORT describes "
                           f"{sorted(CHANGES_IN_SHORT)}")
    after = (f"; and {TIMES[n]} after the held-out run the repository changed without any computation of a frozen "
             "number changing: " + "; ".join(f"the {ORDINALS[k - 1]} time" + ("," if CHANGES_IN_SHORT[k].startswith("after")
                                                                             else "") + f" {CHANGES_IN_SHORT[k]}"
                                              for k in range(1, n + 1)) if n else "")
    return ("The README's section on deviations is the one that `docs/analysis_plan.md`, a frozen file, calls "
            f"\"Deviations and additions\"; the full list is in {link('docs/deviations.md', m)}. In short: two "
            "statements of the "
            "first commit about the data source were withdrawn; the baseline grid was widened after the first commit to "
            "follow the plan text; judge v2 does not do what the plan said it would; the reading of the disagreements "
            "was drafted by the AI coding assistant, not by a person; several analyses were added after DEV results "
            f"were seen{after}. The registered v3 step was not carried out.")


WORDS = {"relief": "money back", "no_relief": "no money back", "cannot_tell": "cannot tell"}
NUMBER_WORDS = ("No", "One", "Two", "Three", "Four", "Five")


def _undo_sentence(ex: dict) -> str:
    cases = (ex.get("undo") or {}).get("cases") or []
    if not cases:
        return ""
    k = len(cases)
    what = "; ".join(f"{WORDS.get(c['undone'], c['undone'])}, then {WORDS.get(c['final'], c['final'])}" for c in cases)
    return (f" {NUMBER_WORDS[k] if k < len(NUMBER_WORDS) else k} answer{'' if k == 1 else 's'} "
            f"{'was' if k == 1 else 'were'} undone and re-entered before the lock ({what}), as the labelling tool "
            "allows; the file keeps every record and the undo, and the last answer counts.")


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
        ex = extra.get(who, {})
        ps = ex.get("paired_sensitivity", {})
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
                   f"; their sum, {r['total_minutes']:.0f} minutes, is therefore not the time spent labelling." +
                   _undo_sentence(ex) + _labelled_anchor(m))
        if "person_vs_recorded" not in f:
            out.append("\nToo few decided cases for agreement figures.")
            continue
        out.append("\n" + person_table({"human": {"annotators": {who: r}}}))
        if not any(v in f["systems_vs_person"] for v in ("v1", "v2")):
            out.append("\nThe judges have not answered these complaints yet (`make judge-labelled`).")
        out += _human_figures(f, ps.get("all"), _sides(ex, "all"))[1:]   # the person's row is in the table above
        out.append("\n" + _cannot_tell(f, r, ex))
    out.append("\n" + _v3(m) + " The author's own reading of each sorted disagreement, from the second screen, is in "
               "the full report.")
    return "\n".join(out)


def _labelled_anchor(m) -> str:
    """The bytes of the judges' answers for the labelled sample, as committed after the lock, and of the locked
    labels, the lock and the second screen, as committed in the lock commit."""
    ig = (m.get("posthoc") or {}).get("integrity") or {}
    files = ig.get("labelled_run_files")
    if not files:
        return ""
    lab = ig.get("label_files") or {}
    return (" The judges' answers for these complaints were added to the response cache after the lock and committed "
            f"in `{ig['labelled_run_commit']}`: " + "; ".join(f"`{k}` {v['bytes']:,} bytes, SHA-256 `{v['sha256'][:16]}...`"
                                                         for k, v in files.items()) +
            "; `make check` fails if these bytes change." +
            (f" The labels, the lock and the second screen as committed in `{ig['labels_commit']}` are anchored the same "
             "way, whole files: " + "; ".join(f"`{k}` {v['bytes']:,} bytes, SHA-256 `{v['sha256'][:16]}...`"
                                             for k, v in lab.items()) +
             "; `make check` fails if any of them changes, even with the lock file resealed to match." if lab else ""))


def _band(b: dict) -> str:
    return (f"under {b['to_s']} s" if not b["from_s"] else f"over {b['from_s']} s" if b["to_s"] is None
            else f"from {b['from_s']} to {b['to_s']} s")


def _undo_in_band(bands: list, cases: list) -> str:
    """Where an answer re-entered after an undo is 'cannot tell', say which time band its recorded time puts it in
    and, when the next band is close, how that band compares without it (descriptive)."""
    out, th = [], ("first", "second", "third", "fourth", "fifth", "sixth")
    for c in cases:
        s = c.get("final_s")
        if s is None or c.get("final") != "cannot_tell":
            continue
        k = next(j for j, b in enumerate(bands) if s >= b["from_s"] and (b["to_s"] is None or s < b["to_s"]))
        if k + 1 >= len(bands) or bands[k]["n"] < 2:
            continue
        b, nx = bands[k], bands[k + 1]
        a, n, r_next = b["cannot_tell"] - 1, b["n"] - 1, nx["cannot_tell"] / nx["n"]
        side = "below" if a / n < r_next else "above" if a / n > r_next else "level with"
        out.append(f" The {th[k]} and {th[k + 1]} bands are close ({R.pct(b['cannot_tell'] / b['n'], 0)} and "
                   f"{R.pct(r_next, 0)}), and one of the {b['cannot_tell']} 'cannot tell' answers {_band(b)} is the answer "
                   f"re-entered after the undo, recorded at {s:.1f} s (the answer it replaced was recorded at "
                   f"{c['undone_s']:.1f} s); without it the {th[k]} band is {a} of {n}, {side} the {th[k + 1]}.")
    return "".join(out)


def _cannot_tell(f, r, ex=None) -> str:
    """What the share of 'cannot tell' means for anyone planning to use labels like these: it describes this one
    annotator. Facts only: the rubric's definition, who the annotator is, the registered count and recorded relief
    rate, the descriptive pace figures (posthoc), and whether reasons were recorded."""
    pc = (ex or {}).get("pace")
    if pc:
        fast = [b for b in pc["bands"] if b["to_s"] is not None and b["to_s"] <= 60]
        slow = [b for b in pc["bands"] if b not in fast]
        split = (f"; that is {sum(b['cannot_tell'] for b in fast)} of the {sum(b['n'] for b in fast)} labels that took "
                 f"under a minute and {sum(b['cannot_tell'] for b in slow)} of the {sum(b['n'] for b in slow)} that took "
                 "longer" if fast and slow else "")
        pace = (" His share of 'cannot tell' by the time spent on a label: " +
                ", ".join(f"{b['cannot_tell']} of {b['n']} {_band(b)}" for b in pc["bands"]) + split + "." +
                _undo_in_band(pc["bands"], ((ex or {}).get("undo") or {}).get("cases") or []) +
                f" His median time per label was {pc['median_s_first']:.0f} s on his first {pc['n_first']} labels "
                f"and {pc['median_s_last']:.0f} s on his last {pc['n_last']} ({AFTER_LOCK}). These figures show how "
                "time and 'cannot tell' went together for this annotator; they cannot separate how much of the share "
                "comes from the texts and how much from his reading and his pace.")
    else:
        pace = ""
    return (f"**'Cannot tell' in {f['cannot_tell']} of {f['n']}.** The rubric asks for 'cannot tell' when the text does "
            "not describe a transaction the reader can identify, or when the decision turns entirely on evidence the "
            "text does not give. This annotator (not a lawyer, not a native English reader, reading with a machine "
            f"translation beside the English, a median of {r['median_seconds']:.0f} s per label) left {f['cannot_tell']} "
            f"of {f['n']} complaints without a decision, so his labels cover {f['decided']} of the {f['n']} cases, and "
            f"which {f['decided']} was chosen by him, not at random (registered: the companies paid in "
            f"{R.pct(f['cannot_tell_recorded_relief'])} of the 'cannot tell' cases, {R.pct(f['recorded_relief_share'])} "
            "of the whole sample)." + pace + " Which of the two reasons applied to a case was not recorded as a field "
            f"({r['with_note']} of {f['n']} labels " + ("carries" if r["with_note"] == 1 else "carry") +
            " a free-text note), so the reasons are not counted.")


def _cannot_tell_rates(g) -> str:
    return (f"'Cannot tell' (registered): {g['cannot_tell']}; recorded relief rate among them "
            f"{R.pct(g['cannot_tell_recorded_relief'])}; relief rate of each system on them: " +
            ", ".join(f"{SHORT.get(n, n)} {R.pct(v)}" for n, v in g["cannot_tell_system_relief"].items()) + ".")


def b_human_detail(m):
    h = m["human"]
    if not h["annotators"]:
        return "_No locked set of labels yet._"
    extra = (m.get("posthoc") or {}).get("human", {})
    out = []
    for who, r in h["annotators"].items():
        f = r["all"]
        ex = extra.get(who, {})
        ps = ex.get("paired_sensitivity", {})
        out.append(f"Annotator `{who}`. Cases answered 'cannot tell' (registered): {f['cannot_tell']}; recorded relief "
                   f"rate among them {R.pct(f['cannot_tell_recorded_relief'])}; relief rate of each system on them: " +
                   ", ".join(f"{SHORT.get(n, n)} {R.pct(v)}" for n, v in f["cannot_tell_system_relief"].items()) +
                   f". Notes written: {r['with_note']}. Cases shown with a flagged translation: "
                   f"{r['translation_flagged_when_shown']}" +
                   (f"; translation shown in {extra[who]['translation_shown']} of {f['n']} ({NOT_REGISTERED})."
                    if who in extra else "."))
        for half, g in r["halves"].items():
            out.append(f"\n{half.capitalize()} half ({g['n']} cases, {g['decided']} decided, {g['cannot_tell']} cannot "
                       "tell). The plan registers these figures for each half; no v3 was run, so nothing is decided from "
                       "them. " + (_cannot_tell_rates(g) if g["cannot_tell"] else ""))
            out += (_human_figures(g, ps.get(half), _sides(ex, half)) if "person_vs_recorded" in g else
                    ["\nToo few decided cases."])
        e = r["error_review"]
        n, t, notes = _went_through(e, _screen(m, who))
        out.append(f"\nSecond screen (after the lock), the author's own reading of the {e['total']} v1 DEV disagreements "
                   f"that the assistant had sorted one by one: he went through {n}{t}" + (f" ({AFTER_LOCK})" if t else "") +
                   f", confirmed {e['confirmed']} drafted readings, changed {len(e['changed'])}{notes}.")
        rows = ex.get("review") or []
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
            "`results/error_analysis/` holds short quotes from DEV narratives. The narratives are redistributed exactly "
            f"as the regulator published them: a check before publication found four of the {dw['records']:,} in which "
            "the regulator's own scrubbing left a name or part of a card number (a consumer's surname, the last four "
            "digits of a card or an account in two, a bank agent's first name and code), and the machine translation "
            "of the one of them in the labelling sample repeats the surname. They are left as published because "
            "`data/records.jsonl.gz` is hashed in the freeze and every result was computed from these texts.")
