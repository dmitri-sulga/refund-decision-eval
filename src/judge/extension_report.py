"""The README sections and full-report sections of the model extension (docs/analysis_plan_models.md, Amendments 1
and 2). Added after both arms had run (eleventh post-run change); presentation only.

Every figure comes from the registered analysis code, run unchanged (models_analysis.analyse and
hosted_analysis.analyse), from the arms' journals and run logs, from the frozen computation (the issuer lookup's
DEV kappa, the person's agreement with the record), or from src/judge/extension_late.py, which computes the
registered outputs that the registered code leaves out and was written after the results were seen (every place
that prints its figures says so). This module computes no metric, interval or label of its own, except the range of
interval widths in the local limits, which it says it computed. `report --check` regenerates
results/models/analysis.json and results/models/hosted/analysis.json exactly as `python -m judge.models_analysis`
and `python -m judge.hosted_analysis` write them, and fails if they differ.
"""
from __future__ import annotations

import json

from . import frontpage as F

R = F.R
LOCAL_HEADING = "Other local models (Amendment 1; added later, exploratory)"
HOSTED_HEADING = "Hosted arm (Amendment 2; added later, exploratory)"   # fixed by Amendment 2, section 10
DETAIL = {"local": "other-local-models-detail", "hosted": "hosted-arm-detail"}   # report.md anchors of the detail
CELL_NAMES = {"v1|dev": "v1, DEV", "v1|labelled": "v1, labelled", "v2|dev": "v2, DEV", "v2|labelled": "v2, labelled"}
LATE_NOTE = ("the code, `src/judge/extension_late.py`, was written after the results were seen; the definitions are "
             "the plan's, unchanged; no label")


def _slug(heading: str) -> str:
    import re
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


# ------------------------------------------------------------------ compute (no model call)
def _jsonl(path) -> list[dict]:
    from .models import _jsonl as read
    return read(path)


def _run_facts(journal: list[dict], run_log: list[dict], order: list[str]) -> dict:
    """Per model and in all: requests journaled by kind, fields left out, run-log events other than a start and a
    finish (counted, and their kinds), and the times of the first and last journal entry."""
    from .models import split_key
    usual = ("run_started", "model_finished")
    per = {}
    for t in order:
        mine = [e for e in journal if split_key(e["key"])[0] == t]
        if not mine:
            continue
        ev = [e for e in run_log if e.get("model") == t]
        per[t] = {"sends": sum(e.get("kind") == "send" for e in mine),
                  "resends": sum(e.get("kind") == "resend" for e in mine),
                  "fields_dropped": sum(bool(e.get("fields_dropped")) for e in mine),
                  "first": min(e["at"] for e in mine), "last": max(e["at"] for e in mine),
                  "runs": sum(e.get("event") == "run_started" for e in ev),
                  "other_event_count": sum(e.get("event") not in usual for e in ev),
                  "other_events": sorted({str(e.get("event")) for e in ev} - set(usual))}
    return {"models": per, "first": min(e["at"] for e in journal) if journal else None,
            "last": max(e["at"] for e in journal) if journal else None,
            "sends": sum(e.get("kind") == "send" for e in journal),
            "resends": sum(e.get("kind") == "resend" for e in journal),
            "fields_dropped": sum(bool(e.get("fields_dropped")) for e in journal),
            "other_event_count": sum(e.get("event") not in usual for e in run_log)}


def compute(m: dict) -> dict:
    """Run the two registered analyses unchanged, the late registered outputs, and collect the run facts."""
    from . import extension_late as L
    from . import hosted as H
    from . import hosted_analysis as HA
    from . import models as X
    from . import models_analysis as MA
    from .data import load_records

    records = load_records()
    local = MA.analyse(records=records)
    hosted = HA.analyse(records=records)
    sets, _ = X.item_sets()
    return {"local": local, "hosted": hosted, "late": L.compute(records, local, hosted),
            "local_runs": _run_facts(_jsonl(X.JOURNAL), _jsonl(X.RUN_LOG), local["order"]),
            "hosted_runs": _run_facts(_jsonl(H.JOURNAL), _jsonl(H.RUN_LOG), hosted["order"]),
            "record_pays_labelled": int(sum(records[i]["label"] for i in sets["labelled"])),
            "n_labelled": len(sets["labelled"])}


def analysis_files(ext: dict) -> list:
    """(path, text) of the two analysis files, written exactly as the registered modules' main() writes them."""
    from . import hosted as H
    from . import models as X
    return [(X.DIR / "analysis.json", json.dumps(ext["local"], indent=1, default=float) + "\n"),
            (H.DIR / "analysis.json", json.dumps(ext["hosted"], indent=1, default=float) + "\n")]


# ------------------------------------------------------------------ formatting
def _n(x, d=3) -> str:
    return R.num(float(x), d)


def _sgn(x, d=3) -> str:
    s = R.num(float(x), d)
    return s if s.startswith("-") else "+" + s


def _paired(iv: dict, level: str = "0.95") -> str:
    lo, hi = iv[level]
    return f"{_sgn(iv['value'])} [{_sgn(lo)}, {_sgn(hi)}]"


def _diff(d: dict) -> str:
    return f"{_sgn(d['value'])} [{_sgn(d['lo'])}, {_sgn(d['hi'])}]"


def _kappa(k: dict) -> str:
    return f"{_n(k['value'])} [{_n(k['lo'])}, {_n(k['hi'])}]"


def _pct(x: dict) -> str:
    return f"{R.pct(x['value'])} [{R.pct(x['lo'])}, {R.pct(x['hi'])}]"


def _pct_level(level: float) -> str:
    """99.643%, 99.286%, 98.75%, 97.5%: the corrected levels as the plan writes them."""
    return f"{100 * level:.3f}".rstrip("0").rstrip(".") + "%"


def _level_key(level: float) -> str:
    return f"{level:.6g}"


def _quote(summary: str) -> str:
    """The registered summary as a block quote, word for word, with a paragraph break before the parts that the
    plan lists separately (the person, the same-vendor paragraph, the cost) so that each can be found."""
    for part in (" The one person who labelled the sample", " Same vendor. ", " Cost: "):
        summary = summary.replace(part, "\n>\n>" + part, 1)
    return "> " + summary


TINY = 1e-9                 # float noise around zero, as in frontpage.py


def _side(k: dict) -> str:
    """How an interval prints against zero. A bound within float noise of zero prints as 0.000 and 'touches zero';
    the registered rule (models_analysis.person_figures) reads such a bound strictly."""
    if abs(k["lo"]) < TINY or abs(k["hi"]) < TINY:
        return "touches zero"
    return "lies above zero" if k["lo"] > 0 else "lies below zero" if k["hi"] < 0 else "includes zero"


def _disagree(an: dict, intervals: bool = True) -> list[str]:
    """The cells whose two registered readings against the person disagree, with both intervals if asked."""
    out = []
    for t in an["order"]:
        for k, c in an["models"][t]["cells"].items():
            p = c.get("vs_person")
            if p and not p["readings_agree"]:
                d, a = p["kappa_decided"], p["kappa_cannot_tell_as_no"]
                if not intervals:
                    out.append(f"`{t}` {CELL_NAMES[k]}")
                    continue
                out.append(f"`{t}` {CELL_NAMES[k]} (on his decided labels {_kappa(d)}, which {_side(d)}; with "
                           f"'cannot tell' as no money back {_kappa(a)}, which {_side(a)})")
    return out


def _corrected(x: dict) -> str:
    """Amendment 1, section 3: every 'above' and 'below' (and 'pays more' or 'pays less') is printed with whether the
    corrected interval gives the same side."""
    if x["also_at_corrected_level"] is None:
        return x["label"]
    return x["label"] + ("; also at the corrected level" if x["also_at_corrected_level"] else
                         "; not at the corrected level")


def _model(t: str, ref: str) -> str:
    return f"`{t}`" + (" (reference, existing answers)" if t == ref else "")


def _hosted_name(ho: dict, t: str) -> str:
    """Amendment 2, section 1: the reference row carries its label wherever it is printed."""
    note = ho["models"][t]["row_note"]
    return f"`{t}`" + (f" ({note})" if note else "")


def _detail(m, arm: str) -> str:
    """The detail section of an arm in results/report.md: from the README, a path; within the report, an anchor."""
    return ("" if m.get(F.LINK_BASE) else "results/report.md") + "#" + DETAIL[arm]


def _utc(at: str) -> str:
    return at[:16].replace("T", " ")


def _plan_text(start: str, end: str) -> str:
    """A passage of docs/analysis_plan_models.md, as the plan has it."""
    from .models import PLAN_FILE
    text = PLAN_FILE.read_text(encoding="utf-8")
    a = text.index(start)
    return text[a:text.index(end, a)].strip()


def _interval_widths(loc: dict) -> tuple[float, float]:
    """Computed here, from the registered intervals: the narrowest and widest 95% interval against the lookup."""
    w = [c["A"]["readings"]["as no money back"]["0.95"][1] - c["A"]["readings"]["as no money back"]["0.95"][0]
         for t in loc["order"] for k, c in loc["models"][t]["cells"].items() if "A" in c]
    return min(w), max(w)


def _person(m) -> dict:
    return F._person_facts(m)[1]["person_vs_recorded"]


def _totals(an: dict, models) -> tuple[int, int]:
    cells = [c for t in models for c in an["models"][t]["cells"].values()]
    return sum(c["answered"] for c in cells), sum(c["malformed"] for c in cells)


def _no_malformed(an: dict) -> bool:
    return all(c["malformed"] == 0 for t in an["order"] for c in an["models"][t]["cells"].values())


def _span(xs: list[int]) -> str:
    xs = sorted(xs)
    return f"{xs[0]} and {xs[1]}" if len(xs) == 2 else f"{xs[0]} to {xs[-1]}"


def _person_line(m, an: dict, intervals: bool = True) -> str:
    pv = _person(m)
    dis = _disagree(an, intervals)
    return (f"One person, the author, labelled these complaints blind; he is the only annotator. On his {pv['n']} "
            f"decided labels he agrees with the record only at kappa {_kappa(pv['kappa'])}, so neither the record "
            "nor this person is a gold standard. " +
            ("The two registered readings against him (his decided labels; all of the sample with 'cannot tell' as "
             "no money back) disagree in sign or side of zero for " + "; ".join(dis) + "; no conclusion is drawn "
             "there" + ("." if intervals else " (both intervals of each are in the full report).") if dis else "The two registered readings against him agree in sign and side of zero in every "
             "cell."))


# ------------------------------------------------------------------ first screen (Amendment 1, section 9; 2, section 10)
def first_screen_sentences(m) -> str:
    """The one sentence each amendment allows on the first screen, with no figure; empty until the arm has answers."""
    ext = m.get("extension")
    if not ext:
        return ""
    base = m.get(F.LINK_BASE, "")
    target = (base + "README.md") if base else ""
    out = []
    if ext["local_runs"]["sends"]:
        out.append("Added later, exploratory: the same two prompts were run on other local models, on DEV and the "
                   f"labelled sample only; see [the section]({target}#{_slug(LOCAL_HEADING)}).")
    if ext["hosted_runs"]["sends"]:
        out.append("Added later, exploratory: the same two prompts were also run on three hosted Anthropic models, on "
                   f"DEV and the labelled sample only; see [the section]({target}#{_slug(HOSTED_HEADING)}).")
    return (" " + " ".join(out)) if out else ""


# ------------------------------------------------------------------ the local arm (README)
def b_other_local_models(m):
    ext = m.get("extension")
    if not ext or not ext["local_runs"]["sends"]:
        return R.PENDING
    loc, runs = ext["local"], ext["local_runs"]
    ref = loc["order"][0]
    new = loc["order"][1:]
    answers, malformed = _totals(loc, new)
    lk = m["dev"]["issuer_rate"]["kappa"]
    lo_w, hi_w = _interval_widths(loc)
    some_a = next(c["A"] for t in new for c in loc["models"][t]["cells"].values() if "A" in c)
    some_b = next(c["B"] for t in new for c in loc["models"][t]["cells"].values() if "B" in c)
    ref_v1 = loc["models"][ref]["cells"]["v1|dev"]["pays"]
    out = [
        "*Added later; exploratory.* Planned after the held-out run, after the blind labels were locked and read and "
        "after every DEV result had been seen. The plan, " + F.link("docs/analysis_plan_models.md", m) + ", was "
        "registered before any of these models was sent a request and amended twice, each time before the requests "
        "the amendment governs; the AI coding assistant chose the models and wrote the plan and the code at the "
        "author's request. The question: do two results of the study belong to the study model or to the task? "
        "They are that neither prompt was above the issuer lookup on DEV, and that judge v1 paid more often than the "
        f"companies did ({ref_v1} of {loc['n_dev']} against {loc['record_pays_dev']}).",
        "",
        "**What was run.** Judge v1 and judge v2, with the frozen prompts and options (thinking off; the repetition, "
        "presence and frequency penalties set to the values the study model ran with, which apply no penalty: "
        f"Amendment 1, section 4), on {len(new)} other local models through Ollama, on DEV ({loc['n_dev']} "
        f"complaints) and the labelled sample ({ext['n_labelled']}) only. No held-out complaint was sent. "
        f"{answers:,} answers, one request each, were sent between {_utc(runs['first'])} and {_utc(runs['last'])} UTC: "
        f"{runs['resends']} resent, {runs['other_event_count']} refusals or interruptions in the run log, "
        f"{runs['fields_dropped']} requests with a field left out, {malformed} malformed "
        f"({R.pct(malformed / answers)} of answers). `{ref}`, the study model, is a reference row: its existing "
        f"answers are shown for comparison (no new call), and it is not counted in the {len(new)} of {len(new)}.",
        "",
        "**The registered summary** (Amendment 1, section 3; the sentence as the registered code builds it):",
        "",
        _quote(loc["summary"]),
        "",
        f"**DEV.** The companies paid {loc['record_pays_dev']} of {loc['n_dev']}. The issuer lookup, which never reads "
        f"the complaint, pays {loc['lookup_pays_dev']} and agrees with the record at kappa {_kappa(lk)}. A paired "
        "interval is the 95% interval of a difference, from resampling the same complaints for both systems. Each "
        "label comes from it. The corrected (Bonferroni) interval is widened for the number of comparisons: "
        f"{_pct_level(some_a['corrected_level'])} for the {2 * len(new)} comparisons with the issuer lookup, "
        f"{_pct_level(some_b['corrected_level'])} for the {len(new)} payout comparisons. \"Also\" or \"not at the "
        "corrected level\" says whether it gives the same side of zero. Where a bound is exactly 0.000, its side of "
        "zero can move with the bootstrap seed, as the human-label section shows for judge v2; the registered code "
        "counts such a bound as including zero.",
        "",
    ]
    rows = []
    for t in loc["order"]:
        for v in ("v1", "v2"):
            c = loc["models"][t]["cells"][f"{v}|dev"]
            a, b = c.get("A"), c.get("B")
            rows.append((_model(t, ref), v, f"{c['pays']} of {c['n']}", _kappa(c["kappa_vs_record"]),
                         _paired(a["readings"]["as no money back"]) if a else "-", _corrected(a) if a else c["status"],
                         (f"pays {c['pays']} of {c['n']}, the companies {loc['record_pays_dev']}: " + _corrected(b))
                         if b else "-"))
    out.append(R.table(("Model", "Prompt", "Pays", "Kappa vs record [95%]", "Kappa minus the issuer lookup [95%]",
                        "Against the issuer lookup", "Payout (v1)"), rows))
    out += [
        "",
        "**Against the person** (the labelled sample; descriptive, no label; per cell in the full report). " +
        _person_line(m, loc, intervals=False),
        "",
        "**Limits.**",
        "",
        "- **Agreement, not correctness.** Every figure is agreement with what the company recorded or with one "
        "person, who agree only weakly with each other.",
        "- **No ranking.** Each model is compared with the issuer lookup and with the record, not with the other "
        "models; no model is called the best.",
        f"- **DEV favours the study model.** v2 was written from `{ref}`'s DEV disagreements, so on the other models "
        "it is a prompt tuned for a different model; the issuer lookup's threshold was chosen after DEV.",
        f"- **Small samples.** The 95% paired intervals against the issuer lookup are {_n(lo_w, 2)} to "
        f"{_n(hi_w, 2)} wide (computed by the presentation code from the registered intervals). What the design can "
        "detect is in the full report.",
        "- **One run of one build per model**, under one Ollama version on one laptop. Runner and quantisation go "
        "with the model, so no difference here is one of family or size alone, and asking again can give other "
        "answers. Only models that fit this laptop with reasoning off are covered. Any model may have seen these "
        f"complaints and their outcomes in training. The held-out verdicts stay those of `{ref}`.",
        "",
        "In the full report: the run record; per cell, malformed answers, seconds per call, agreement with the study "
        "model and the figures against the person; the corrected intervals; what the design can detect; and, "
        "marked as computed after the results were seen, the registered outputs the registered code leaves out "
        f"([Other local models detail]({_detail(m, 'local')})).",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ the hosted arm (README)
def _escape_cells(table: str) -> str:
    """hosted_analysis.table_markdown writes the cell as 'v1|dev', whose '|' splits a Markdown table cell; the
    character is escaped here, so the table renders with the columns the code intends (recorded as a deviation)."""
    for k in CELL_NAMES:
        table = table.replace(f"| {k} |", "| " + k.replace("|", "\\|") + " |")
    return table


def b_hosted_arm(m):
    ext = m.get("extension")
    if not ext or not ext["hosted_runs"]["sends"]:
        return R.PENDING
    ho, runs, loc = ext["hosted"], ext["hosted_runs"], ext["local"]
    from . import hosted as H
    answers, malformed = _totals(ho, ho["order"])
    cells = [c for t in ho["order"] for c in ho["models"][t]["cells"].values()]
    routed = sum(c["routed_elsewhere"] for c in cells)
    other_stops = sum(n for c in cells for s, n in c["stop_reasons"].items() if s != "end_turn")
    compared = [t for t in ho["order"] if ho["models"][t]["compared"]]
    refs = [t for t in ho["order"] if not ho["models"][t]["compared"]]
    la, lb = 1 - 0.05 / ho["comparisons"]["A"], 1 - 0.05 / ho["comparisons"]["B"]
    pv = _person(m)
    clean = not (runs["resends"] or runs["other_event_count"] or other_stops or malformed)
    out = [
        "*Added later; exploratory.* Registered as Amendment 2 of the same plan before any request to these models. "
        f"Judge v1 and judge v2, with the frozen prompts, ran on {len(ho['order'])} hosted Anthropic models through "
        f"the Claude API, on the same DEV ({ho['n_dev']}) and labelled ({ext['n_labelled']}) complaints. Settings: API "
        f"version `{H.API_VERSION}`, reasoning effort low, no temperature (this API version rejects it) and no seed. "
        f"{answers:,} answers, one request each, were sent between {_utc(runs['first'])} and {_utc(runs['last'])} UTC. "
        + ("None was resent, failed, cut short or refused, and none was malformed "
           f"({R.pct(malformed / answers)})" if clean else
           f"{runs['resends']} were resent, {runs['other_event_count']} run-log events were errors or interruptions, "
           f"{other_stops} answers were cut short or refused, and {malformed} were malformed "
           f"({R.pct(malformed / answers)})") +
        (f"; all {answers:,} were served through global routing" if not routed else
         f"; {routed} were served through other than global routing") + ". The full run record is in the report.",
        "",
        " and ".join(f"`{t}`" for t in compared) + " ran without up-front thinking (no reasoning step before the "
        "answer) and form the comparison set. " + " and ".join(f"`{t}`" for t in refs) + " cannot have its reasoning "
        "turned off, so it is a reference row: shown, not compared, not counted. Status: " +
        "; ".join(f"`{t}` {ho['models'][t]['status']}" for t in ho["order"]) + ".",
        "",
        f"In the table, \"Pays\" is out of {ho['n_dev']} on DEV, where the companies paid {ho['record_pays_dev']}, and "
        f"out of {ext['n_labelled']} on the labelled sample, where they paid {ext['record_pays_labelled']}. Kappa "
        f"against the record on the labelled sample is over all {ext['n_labelled']} complaints; against the person, "
        f"over his {pv['n']} decided labels. The annotator read each complaint beside a machine translation made by "
        "`claude-fable-5-2`, a model of the same vendor, so the hosted models' agreement with him is not independent "
        "of the vendor.",
        "",
        _escape_cells(ho["table"]),
        "",
        "**The registered summary** (Amendment 2, section 10; as the registered code builds it, with the sentence on "
        "the reference row, the same-vendor disclosure and the cost):",
        "",
        _quote(ho["summary"]),
        "",
        "*Not registered:* the \"Same vendor\" paragraph is quoted word for word from the plan as registered before "
        "the run, which is why it speaks of the run as still to come; the reviews it calls independent were "
        "separate AI-agent sessions, not people. \"Adaptive thinking\" means that the model decides whether, and how "
        "much, to reason before it answers.",
        "",
        f"**The intervals behind the labels** (DEV). Corrected (Bonferroni) levels for this arm: {_pct_level(la)} for "
        f"the {ho['comparisons']['A']} comparisons with the issuer lookup, {_pct_level(lb)} for the "
        f"{ho['comparisons']['B']} payout comparisons. The reference row gets the 95% intervals only, with no label.",
        "",
    ]
    rows = []
    for t in ho["order"]:
        mm = ho["models"][t]
        for v in ("v1", "v2"):
            c = mm["cells"][f"{v}|dev"]
            if mm["compared"]:
                a, b = c["A"], c.get("B")
                ra = a["readings"]["as no money back"]
                rb = b["readings"]["as no money back"] if b else None
                rows.append((_hosted_name(ho, t), v, _paired(ra), _paired(ra, _level_key(la)), _corrected(a),
                             _paired(rb) if b else "-", _paired(rb, _level_key(lb)) if b else "-",
                             (f"pays {c['pays']} of {c['n']}, the companies {ho['record_pays_dev']}: " + _corrected(b))
                             if b else "-"))
            else:
                b = c.get("B_descriptive")
                rows.append((_hosted_name(ho, t), v, _paired(c["A_descriptive"]), "-", "descriptive only",
                             _paired(b) if b else "-", "-",
                             f"pays {c['pays']} of {c['n']}; descriptive only" if b else "-"))
    out.append(R.table(("Model", "Prompt", "Kappa minus the issuer lookup, 95%", f"At {_pct_level(la)}",
                        "Against the issuer lookup", "Payout share minus the companies', 95%", f"At {_pct_level(lb)}",
                        "Payout (v1)"), rows))
    dis = _disagree(ho)
    v1 = {t: loc["models"][t]["cells"]["v1|dev"]["pays"] for t in loc["order"]}
    v2 = {t: loc["models"][t]["cells"]["v2|dev"]["pays"] for t in loc["order"]}
    ref = loc["order"][0]
    l1, l2 = [v1[t] for t in loc["order"][1:]], [v2[t] for t in loc["order"][1:]]
    h1 = [ho["models"][t]["cells"]["v1|dev"]["pays"] for t in compared]
    h2 = [ho["models"][t]["cells"]["v2|dev"]["pays"] for t in compared]
    r1 = "; ".join(f"`{t}` {ho['models'][t]['cells']['v1|dev']['pays']}" for t in refs)
    r2 = "; ".join(f"`{t}` {ho['models'][t]['cells']['v2|dev']['pays']}" for t in refs)
    out += [
        "",
        "**Against the person** (the labelled sample; descriptive, no label). The table gives his decided labels; "
        "the second registered reading, with 'cannot tell' counted as no money back, is in the full report. " +
        (("The two readings disagree for " + "; ".join(dis) + "; no conclusion is drawn there.") if dis else
         "The two readings agree in sign and side of zero in every cell."),
        "",
        "**Limits.** Those of the local models apply. In addition, no temperature or seed could be sent, so each "
        "figure is one sample of each model as the provider served it on that date; the provider's format "
        "instructions and tokenizer also differ from the local runs. The two compared tiers differ "
        "in more than size, and the vendor of these models also made the tools that built this arm (above).",
        "",
        "**Exploratory (not registered).** Looked at after the results were seen; descriptive counts only, with no "
        "test and no interval. The arms are listed side by side here, not compared, and the reference rows are "
        f"given apart. On DEV, judge v1 paid out on {_span(l1)} of the {loc['n_dev']} complaints on the {len(l1)} other "
        f"local models and on {_span(h1)} on the {len(h1)} compared hosted models (reference rows: `{ref}` "
        f"{v1[ref]}; {r1}); the companies paid out on {loc['record_pays_dev']} and the issuer lookup on "
        f"{loc['lookup_pays_dev']}. Judge v2 paid out on {_span(l2)} on the local models and on {_span(h2)} on the "
        f"compared hosted models (reference rows: `{ref}` {v2[ref]}; {r2}).",
        "",
        "In the full report: the run record; per cell, seconds per call, agreement with the study model and the "
        "second reading against the person; and, marked as computed after the results were seen, the registered "
        f"outputs the registered code leaves out ([Hosted arm detail]({_detail(m, 'hosted')})).",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ the full report only
def _late_table(ext: dict, arm: str, name) -> str:
    late = ext["late"]["arms"][arm]
    an = ext[arm]
    rows = []
    for t in an["order"]:
        for k, f in late.get(t, {}).items():
            rows.append((name(t), CELL_NAMES[k], _pct(f["accuracy_vs_record"]),
                         f"{f['pays']} of {f['n']} (the companies {f['record_pays']}): "
                         f"{_paired(f['payout_minus_record'])}",
                         _diff(f["minus_tfidf"]["kappa"]) if "minus_tfidf" in f else "-",
                         _pct(f["accuracy_vs_person_decided"]) if "accuracy_vs_person_decided" in f else "-",
                         _pct(f["accuracy_vs_person_cannot_tell_as_no"])
                         if "accuracy_vs_person_cannot_tell_as_no" in f else "-"))
    n_dec = next((f["accuracy_vs_person_decided"]["n"] for t in late for f in late[t].values()
                  if "accuracy_vs_person_decided" in f), "")
    return R.table(("Model", "Cell", "Accuracy vs record [95%]", "Payout share minus the record's [95%]",
                    "Judge minus TF-IDF, kappa [95%] (DEV)", f"Accuracy vs the person, {n_dec} decided [95%]",
                    f"Accuracy vs the person, all {ext['n_labelled']}, 'cannot tell' as no money back [95%]"), rows)


def _late_head(ext: dict) -> str:
    tk = ext["late"]["tfidf_dev_kappa"]
    return (f"**Registered outputs computed after the results were seen** ({LATE_NOTE}). Plan items 1 to 4: accuracy "
            "against the record, with the study's bootstrap; the payout share minus the record's for every cell, "
            "with its paired interval; on DEV, judge minus TF-IDF, where TF-IDF is the study's own model with its "
            f"frozen settings (DEV kappa {_kappa(tk)}); and accuracy against the person under the two registered "
            "readings. Item 5's tokens per second are not computed: the plan does not fix how per-call rates are "
            "aggregated, and the hosted answers carry no server timings. The list of resends that Amendment 1, "
            "section 5 promises is empty: the journals hold no resend.")


def b_other_local_models_detail(m):
    ext = m.get("extension")
    if not ext or not ext["local_runs"]["sends"]:
        return R.PENDING
    loc, runs = ext["local"], ext["local_runs"]
    ref = loc["order"][0]
    pv = _person(m)
    out = ["Added later; exploratory. Every figure below comes from `src/judge/models_analysis.py`, run unchanged "
           "(it writes `results/models/analysis.json`), or from the journal and run log of the run, except the "
           "table marked as computed after the results were seen.", "",
           "**Run record** (from `results/models/sent.jsonl` and `results/models/run_log.jsonl`):", ""]
    out.append(R.table(("Model", "Runs", "Requests sent", "Resent", "Fields left out", "First request (UTC)",
                        "Last request (UTC)", "Status", "Other run-log events"),
                       [(f"`{t}`", r["runs"], r["sends"], r["resends"], r["fields_dropped"], _utc(r["first"]),
                         _utc(r["last"]), loc["models"][t]["status"].replace("model_finished", "finished"),
                         f"{r['other_event_count']}" + (f" ({', '.join(r['other_events'])})" if r["other_events"]
                                                        else "")) for t, r in runs["models"].items()]))
    out += ["", "**Per cell.** Malformed answers include any `done_reason` other than `stop` and any prompt over the "
            "context limit (Amendment 1, section 7). Seconds per call were measured on a laptop used for other work "
            "and are not compared between models. Agreement with the study model is kappa between the two models' "
            "decisions on the same cases (descriptive).", ""]
    rows = []
    for t in loc["order"]:
        for k, c in loc["models"][t]["cells"].items():
            s = c.get("seconds_per_call")
            rows.append((_model(t, ref), CELL_NAMES[k], f"{c['answered']} of {c['n']}",
                         f"{c['malformed']} ({R.pct(c['malformed'] / c['n'])})", c["pays"],
                         _kappa(c["kappa_vs_record"]),
                         _n(c["kappa_with_reference"]) if c.get("kappa_with_reference") is not None else "-",
                         f"{s['median']:.1f} / {s['mean']:.1f}" if s else "-"))
    out.append(R.table(("Model", "Cell", "Answered", "Malformed (share)", "Pays", "Kappa vs record [95%]",
                        f"Kappa with `{ref}`", "Seconds per call (median / mean)"), rows))
    out += ["", f"**The labelled sample** (descriptive; no label). The companies paid {ext['record_pays_labelled']} "
            f"of {ext['n_labelled']}. " + _person_line(m, loc), ""]
    rows = []
    for t in loc["order"]:
        for v in ("v1", "v2"):
            c = loc["models"][t]["cells"][f"{v}|labelled"]
            p = c["vs_person"]
            rows.append((_model(t, ref), v, f"{c['pays']} of {c['n']}", _kappa(c["kappa_vs_record"]),
                         _kappa(p["kappa_decided"]), _kappa(p["kappa_cannot_tell_as_no"]),
                         "agree" if p["readings_agree"] else "disagree: no conclusion"))
    out.append(R.table(("Model", "Prompt", "Pays", f"Kappa vs record, all {ext['n_labelled']} [95%]",
                        f"Kappa vs the person, {pv['n']} decided [95%]",
                        f"Kappa vs the person, all {ext['n_labelled']}, 'cannot tell' as no money back [95%]",
                        "The two readings"), rows))
    some_a = next(c["A"] for t in loc["order"] for c in loc["models"][t]["cells"].values() if "A" in c)
    some_b = next(c["B"] for t in loc["order"] for c in loc["models"][t]["cells"].values() if "B" in c)
    la, lb = _level_key(some_a["corrected_level"]), _level_key(some_b["corrected_level"])
    out += ["", f"**The intervals behind the DEV labels**, at 95% and at the corrected levels "
            f"({_pct_level(some_a['corrected_level'])} and {_pct_level(some_b['corrected_level'])}):", ""]
    rows = []
    for t in loc["order"]:
        for v in ("v1", "v2"):
            c = loc["models"][t]["cells"][f"{v}|dev"]
            a = c["A"]["readings"]["as no money back"]
            b = c["B"]["readings"]["as no money back"] if "B" in c else None
            rows.append((_model(t, ref), v, _paired(a), _paired(a, la), a["corrected_label"],
                         _paired(b) if b else "-", _paired(b, lb) if b else "-", b["corrected_label"] if b else "-"))
    out.append(R.table(("Model", "Prompt", "Kappa minus the lookup, 95%", "Corrected", "Label at the corrected level",
                        "Payout share minus the record's, 95%", "Corrected", "Label at the corrected level"), rows))
    fs = [f"{t} {k}" for t in loc["order"] for k, c in loc["models"][t]["cells"].items()
          for x in ("A", "B") if c.get(x, {}).get("format_sensitive")]
    out += ["", ("**Three readings of malformed answers** (Amendment 1, section 7). " +
                 ("No answer was malformed, so the three readings score the same answers and give the same labels; no "
                  "cell is format-sensitive." if _no_malformed(loc) and not fs else
                  "Format-sensitive cells: " + (", ".join(fs) or "none") + "."))]
    out += ["", _late_head(ext), "", _late_table(ext, "local", lambda t: _model(t, ref))]
    out += ["", "**What the design can detect.** Amendment 1, section 2, printed with the results as the plan has it:",
            "", _plan_text("- **Source.**", "**§3 The summary sentence**"),
            "", "**Speculative decoding**, left as shipped (Amendment 1, section 4, as the plan records it; not measured "
            "per answer):", "", _plan_text("- **Speculative decoding**", "**§5 ")]
    out += ["", "**Where the registered code and the plan part.** Each is recorded as a deviation in " +
            F.link("docs/deviations.md", m) + ":", "",
            "- accuracy (items 1 and 4), judge minus TF-IDF (item 2) and the payout interval outside v1 on DEV (item "
            "3) are not computed by the registered code; they are computed above, after the results were seen, with "
            "the plan's definitions;",
            "- tokens per second (item 5) are not computed, for the reason given above;",
            "- the registered code lists no resend and states no request field left out (Amendment 1, section 5); "
            "both lists are empty for this run, as the run record shows."]
    return "\n".join(out)


def b_hosted_arm_detail(m):
    ext = m.get("extension")
    if not ext or not ext["hosted_runs"]["sends"]:
        return R.PENDING
    ho, runs = ext["hosted"], ext["hosted_runs"]
    ref = ho["reference"]
    out = ["Added later; exploratory. Every figure below comes from `src/judge/hosted_analysis.py`, run unchanged "
           "(it writes `results/models/hosted/analysis.json`), or from the arm's journal, run log and spend journal, "
           "except the table marked as computed after the results were seen.",
           "", "**Run record** (from `results/models/hosted/sent.jsonl` and `results/models/hosted/run_log.jsonl`):", ""]
    out.append(R.table(("Model", "Runs", "Requests sent", "Resent", "First request (UTC)", "Last request (UTC)",
                        "Status", "Other run-log events"),
                       [(_hosted_name(ho, t), r["runs"], r["sends"], r["resends"], _utc(r["first"]), _utc(r["last"]),
                         ho["models"][t]["status"],
                         f"{r['other_event_count']}" + (f" ({', '.join(r['other_events'])})" if r["other_events"]
                                                        else "")) for t, r in runs["models"].items()]))
    out += ["", "**Per cell**, the figures the registered table does not print: the malformed share, seconds per call "
            "(the first call per prompt version includes compiling the schema), agreement with the study model on "
            "the same cases (descriptive), and the second registered reading against the person ('cannot tell' "
            "counted as no money back, all of the labelled sample). Where the two readings fall on different sides "
            "of zero, or have different signs, no conclusion is drawn.", ""]
    rows = []
    for t in ho["order"]:
        for k, c in ho["models"][t]["cells"].items():
            s = c.get("seconds_per_call")
            p = c.get("vs_person")
            rows.append((_hosted_name(ho, t), CELL_NAMES[k], R.pct(c["malformed"] / c["n"]),
                         f"{s['median']:.1f} / {s['mean']:.1f}" if s else "-", _n(c["kappa_with_reference"]),
                         _kappa(p["kappa_cannot_tell_as_no"]) if p else "-",
                         ("agree" if p["readings_agree"] else "disagree: no conclusion") if p else "-"))
    out.append(R.table(("Model", "Cell", "Malformed share", "Seconds per call (median / mean)", f"Kappa with `{ref}`",
                        f"Kappa vs the person, all {ext['n_labelled']}, 'cannot tell' as no money back [95%]",
                        "The two readings"), rows))
    out += ["", _late_head(ext), "", _late_table(ext, "hosted", lambda t: _hosted_name(ho, t))]
    out += ["", "**What the registered table and summary do not print, and where it is here.** Each is recorded as a "
            "deviation in " + F.link("docs/deviations.md", m) + ":", "",
            "- the 95% and corrected intervals behind the labels and the corrected-level words (Amendment 2, "
            "section 10, \"printed as in Amendment 1, section 3\"; plan line 111, \"Bonferroni intervals are printed "
            "beside them\"), and the reference row's 95% paired intervals against the issuer lookup and the record "
            "(section 3): computed by the registered code, printed in the README section under the registered "
            "summary;",
            "- each model's status when it is \"complete\" (section 10): the table names a status only when it is "
            "not; printed in the README section;",
            "- the registered table writes each cell as `v1|dev`, and the `|` splits a Markdown table cell; the "
            "README escapes that one character and prints the table otherwise as the code writes it;",
            "- accuracy, judge minus TF-IDF and the payout interval outside v1 on DEV: computed above, after the "
            "results were seen, with the plan's definitions; tokens per second: not computed, as for the local arm;",
            "- no resend, no lost answer and no waited-out error occurred, so the registered list of resends and "
            "charges without an answer is the sentence \"No request was resent or charged without an answer.\""]
    return "\n".join(out)
