"""The README sections and full-report sections of the model extension (docs/analysis_plan_models.md, Amendments 1
and 2). Added after both arms had run (eleventh post-run change); presentation only.

Every figure comes from the registered analysis code, run unchanged (models_analysis.analyse and
hosted_analysis.analyse), from the arms' journals and run logs, or from the frozen computation (the issuer lookup's
DEV kappa, the person's agreement with the record). Nothing here computes a new metric, interval or label: it prints
what the registered code computed, including the parts of it that the registered code computes but does not print
(the intervals behind the labels, the corrected-level check, the reference row's intervals). `report --check`
regenerates results/models/analysis.json and results/models/hosted/analysis.json exactly as
`python -m judge.models_analysis` and `python -m judge.hosted_analysis` write them, and fails if they differ.
"""
from __future__ import annotations

import json

from . import frontpage as F

R = F.R
LOCAL_HEADING = "Other local models (Amendment 1; added later, exploratory)"
HOSTED_HEADING = "Hosted arm (Amendment 2; added later, exploratory)"   # fixed by Amendment 2, section 10
DETAIL = {"local": "other-local-models-detail", "hosted": "hosted-arm-detail"}   # report.md anchors of the detail
PROMPTS = {"v1": "v1", "v2": "v2"}
CELL_NAMES = {"v1|dev": "v1, DEV", "v1|labelled": "v1, labelled", "v2|dev": "v2, DEV", "v2|labelled": "v2, labelled"}


def _slug(heading: str) -> str:
    import re
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


# ------------------------------------------------------------------ compute (no model call)
def _jsonl(path) -> list[dict]:
    from .models import _jsonl as read
    return read(path)


def _run_facts(journal: list[dict], run_log: list[dict], order: list[str]) -> dict:
    """Per model and in all: requests journaled by kind, fields left out, run log events other than a start and a
    finish, and the times of the first and last journal entry. Read from the arm's journal and run log."""
    from .models import split_key
    per = {}
    for t in order:
        mine = [e for e in journal if split_key(e["key"])[0] == t]
        if not mine:
            continue
        ev = [e for e in run_log if e.get("model") == t]
        per[t] = {"sends": sum(e.get("kind") == "send" for e in mine),
                  "resends": sum(e.get("kind") == "resend" for e in mine),
                  "other_journal_kinds": sorted({str(e.get("kind")) for e in mine} - {"send", "resend"}),
                  "fields_dropped": sum(bool(e.get("fields_dropped")) for e in mine),
                  "first": min(e["at"] for e in mine), "last": max(e["at"] for e in mine),
                  "runs": sum(e.get("event") == "run_started" for e in ev),
                  "other_events": sorted({e.get("event") for e in ev} - {"run_started", "model_finished"})}
    return {"models": per, "first": min(e["at"] for e in journal) if journal else None,
            "last": max(e["at"] for e in journal) if journal else None,
            "sends": sum(e.get("kind") == "send" for e in journal),
            "resends": sum(e.get("kind") == "resend" for e in journal),
            "fields_dropped": sum(bool(e.get("fields_dropped")) for e in journal),
            "other_events": sorted({e.get("event") for e in run_log} - {"run_started", "model_finished"})}


def compute(m: dict) -> dict:
    """Run the two registered analyses unchanged and collect the run facts. Calls no model."""
    from . import hosted as H
    from . import hosted_analysis as HA
    from . import models as X
    from . import models_analysis as MA
    from .data import load_records

    records = load_records()
    local = MA.analyse(records=records)
    hosted = HA.analyse(records=records)
    sets, _ = X.item_sets()
    return {"local": local, "hosted": hosted,
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


def _kappa(k: dict) -> str:
    return f"{_n(k['value'])} [{_n(k['lo'])}, {_n(k['hi'])}]"


def _pct_level(level: float) -> str:
    """99.643%, 99.286%, 98.75%, 97.5%: the corrected levels as the plan writes them."""
    return f"{100 * level:.3f}".rstrip("0").rstrip(".") + "%"


def _quote(summary: str) -> str:
    """The registered summary as a block quote, word for word, with a paragraph break before the parts that the
    plan lists separately (the person, the same-vendor paragraph, the cost) so that each can be found."""
    for part in (" The one person who labelled the sample", " Same vendor. ", " Cost: "):
        summary = summary.replace(part, "\n>\n>" + part, 1)
    return "> " + summary


def _disagree(an: dict) -> list[str]:
    return [f"`{t}` {CELL_NAMES[k]}" for t in an["order"] for k, c in an["models"][t]["cells"].items()
            if "vs_person" in c and not c["vs_person"]["readings_agree"]]


def _corrected(x: dict) -> str:
    """Amendment 1, section 3: every 'above' and 'below' (and 'pays more' or 'pays less') is printed with whether the
    corrected interval gives the same side."""
    if x["also_at_corrected_level"] is None:
        return x["label"]
    return x["label"] + ("; also at the corrected level" if x["also_at_corrected_level"] else
                         "; not at the corrected level")


def _model(t: str, ref: str) -> str:
    return f"`{t}`" + (" (reference, existing answers)" if t == ref else "")


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
    w = [c["A"]["readings"]["as no money back"]["0.95"][1] - c["A"]["readings"]["as no money back"]["0.95"][0]
         for t in loc["order"] for k, c in loc["models"][t]["cells"].items() if "A" in c]
    return min(w), max(w)


def _person(m) -> dict:
    return F._person_facts(m)[1]["person_vs_recorded"]["kappa"]


def _no_malformed(an: dict) -> bool:
    return all(c["malformed"] == 0 for t in an["order"] for c in an["models"][t]["cells"].values())


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
    answers = sum(c["answered"] for t in new for c in loc["models"][t]["cells"].values())
    malformed = sum(c["malformed"] for t in new for c in loc["models"][t]["cells"].values())
    lk = m["dev"]["issuer_rate"]["kappa"]
    pk = _person(m)
    lo_w, hi_w = _interval_widths(loc)
    some = next(c for t in new for c in loc["models"][t]["cells"].values() if "A" in c)
    lv_a = _pct_level(some["A"]["corrected_level"])
    lv_b = _pct_level(next(c for t in new for c in loc["models"][t]["cells"].values() if "B" in c)["B"]
                      ["corrected_level"])
    out = [
        "*Added later; exploratory.* Planned after the held-out run, after the blind labels were locked and read and "
        "after every DEV result had been seen. The plan, " + F.link("docs/analysis_plan_models.md", m) + ", was "
        "registered before any of these models was sent a request and amended twice, each time before the requests "
        "the amendment governs; the AI coding assistant chose the models and wrote the plan and the code at the "
        "author's request. The question: do two results of the study belong to the study model or to the task? "
        "They are that neither prompt beats the issuer lookup on DEV, and that judge v1 pays far more often than the "
        "companies did.",
        "",
        f"**What was run.** Judge v1 and judge v2, with the frozen prompts and options (thinking off, plus the neutral "
        f"sampling penalties of Amendment 1, section 4), on {len(new)} other local models through Ollama, on DEV ({loc['n_dev']} complaints) and the labelled sample ({ext['n_labelled']}) "
        f"only. No held-out complaint was sent. {answers:,} answers, one request each, sent between {_utc(runs['first'])} "
        f"and {_utc(runs['last'])} UTC: {runs['resends']} resent, {len(runs['other_events'])} refusals or interruptions "
        f"in the run log, {runs['fields_dropped']} requests with a field left out, {malformed} malformed. "
        f"`{ref}`, the study model, is the reference row, from its existing answers (no new call).",
        "",
        "**The registered summary** (Amendment 1, section 3; the sentence as the registered code builds it):",
        "",
        _quote(loc["summary"]),
        "",
        f"**DEV.** The companies paid {loc['record_pays_dev']} of {loc['n_dev']}. The issuer lookup, which never reads "
        f"the complaint, pays {loc['lookup_pays_dev']} and agrees with the record at kappa {_kappa(lk)}. Each label "
        f"comes from the 95% paired interval. \"Also\" or \"not at the corrected level\" says whether the Bonferroni "
        f"interval gives the same side: {lv_a} for the {2 * len(new)} comparisons with the issuer lookup, {lv_b} for "
        f"the {len(new)} payout comparisons.",
        "",
    ]
    rows = []
    for t in loc["order"]:
        for v in ("v1", "v2"):
            c = loc["models"][t]["cells"][f"{v}|dev"]
            a = c.get("A")
            b = c.get("B")
            rows.append((_model(t, ref), v, f"{c['pays']} of {c['n']}", _kappa(c["kappa_vs_record"]),
                         _paired(a["readings"]["as no money back"]) if a else "-", _corrected(a) if a else c["status"],
                         (f"pays {c['pays']} of {c['n']}, the companies {loc['record_pays_dev']}: " + _corrected(b))
                         if b else "-"))
    out.append(R.table(("Model", "Prompt", "Pays", "Kappa vs record [95%]", "Kappa minus the issuer lookup [95%]",
                        "Against the issuer lookup", "Payout (v1)"), rows))
    rec_lab = ext["record_pays_labelled"]
    out += [
        "",
        f"**The labelled sample** (descriptive; no label). The companies paid {rec_lab} of {ext['n_labelled']}. "
        f"One person, the author, labelled these complaints blind, and he is the only annotator. On his "
        f"{pk_n(m)} decided labels he agrees with the record only at kappa {_kappa(pk)}, an interval that includes "
        "zero, so neither the record nor this person is a gold standard. The two registered readings against him are "
        "his decided labels only, and all of them with 'cannot tell' counted as no money back; where they fall on "
        "different sides of zero, or have different signs, no conclusion is drawn.",
        "",
    ]
    rows = []
    for t in loc["order"]:
        for v in ("v1", "v2"):
            c = loc["models"][t]["cells"][f"{v}|labelled"]
            p = c["vs_person"]
            rows.append((_model(t, ref), v, f"{c['pays']} of {c['n']}", _kappa(c["kappa_vs_record"]),
                         _kappa(p["kappa_decided"]), _kappa(p["kappa_cannot_tell_as_no"]),
                         "agree" if p["readings_agree"] else "disagree: no conclusion"))
    out.append(R.table(("Model", "Prompt", "Pays", "Kappa vs record [95%]",
                        f"Kappa vs the person, {pk_n(m)} decided [95%]",
                        "Kappa vs the person, 'cannot tell' as no money back [95%]", "The two readings"), rows))
    out += [
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
        f"{_n(hi_w, 2)} wide. \"Not distinguishable\" does not mean \"no better than the lookup\" (what the design "
        "can detect is in the full report).",
        "- **One run of one build per model**, under one Ollama version on one laptop. Runner and quantisation go "
        "with the model, so no difference here is one of family or size alone, and asking again can give other "
        "answers. Only models that fit this laptop with reasoning off are covered. Any model may have seen these "
        f"complaints and their outcomes in training. The held-out verdicts stay those of `{ref}`.",
        "",
        "Per cell, in the full report: malformed answers, seconds per call, agreement with the study model, the "
        "corrected intervals, the run record, what the design can detect, and the registered outputs that the code "
        f"does not produce ([Other local models detail]({_detail(m, 'local')})).",
    ]
    return "\n".join(out)


def pk_n(m) -> int:
    return F._person_facts(m)[1]["person_vs_recorded"]["n"]


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
    answers = sum(c["answered"] for t in ho["order"] for c in ho["models"][t]["cells"].values())
    malformed = sum(c["malformed"] for t in ho["order"] for c in ho["models"][t]["cells"].values())
    routed = sum(c["routed_elsewhere"] for t in ho["order"] for c in ho["models"][t]["cells"].values())
    other_stops = sum(n for t in ho["order"] for c in ho["models"][t]["cells"].values()
                      for s, n in c["stop_reasons"].items() if s != "end_turn")
    compared = [t for t in ho["order"] if ho["models"][t]["compared"]]
    refs = [t for t in ho["order"] if not ho["models"][t]["compared"]]
    lv_a = _pct_level(1 - 0.05 / ho["comparisons"]["A"])
    lv_b = _pct_level(1 - 0.05 / ho["comparisons"]["B"])
    out = [
        "*Added later; exploratory.* Registered as Amendment 2 of the same plan before any request to these models. "
        f"Judge v1 and judge v2, with the frozen prompts, on {len(ho['order'])} hosted Anthropic models through the "
        f"Claude API (version `{H.API_VERSION}`, effort low, no sampling parameters, which this API version rejects), on the same DEV ({ho['n_dev']}) and labelled ({ext['n_labelled']}) complaints. {answers:,} answers, "
        f"one request each, sent between {_utc(runs['first'])} and {_utc(runs['last'])} UTC: {runs['resends']} resent, "
        f"{len(runs['other_events'])} error answers or interruptions in the run log, {other_stops} answers that stopped "
        f"other than at the end of their turn (a refusal would be one), {malformed} malformed, {routed} routed other "
        "than globally. " + " and ".join(f"`{t}`" for t in compared) + " ran without "
        "up-front thinking and form the comparison set. " + " and ".join(f"`{t}`" for t in refs) + " cannot have its "
        "reasoning turned off, so it is a reference row: shown, not compared, not counted. Status: " +
        "; ".join(f"`{t}` {ho['models'][t]['status']}" for t in ho["order"]) + ".",
        "",
        _escape_cells(ho["table"]),
        "",
        "**The registered summary** (Amendment 2, section 10; as the registered code builds it, with the sentence on "
        "the reference row, the same-vendor disclosure and the cost):",
        "",
        _quote(ho["summary"]),
        "",
        f"**The intervals behind the labels** (DEV; the companies paid {ho['record_pays_dev']} of {ho['n_dev']}, the "
        f"issuer lookup {ho['lookup_pays_dev']}). Corrected levels for this arm: {lv_a} for the "
        f"{ho['comparisons']['A']} comparisons with the issuer lookup, {lv_b} for the {ho['comparisons']['B']} payout "
        "comparisons. The reference row gets the 95% intervals only, with no label.",
        "",
    ]
    dis = _disagree(ho)
    rows = []
    for t in ho["order"]:
        mm = ho["models"][t]
        for v in ("v1", "v2"):
            c = mm["cells"][f"{v}|dev"]
            if mm["compared"]:
                a, b = c["A"], c.get("B")
                rows.append((f"`{t}`", v, _paired(a["readings"]["as no money back"]), _corrected(a),
                             _paired(b["readings"]["as no money back"]) if b else "-",
                             (f"pays {c['pays']} of {c['n']}, the companies {ho['record_pays_dev']}: " + _corrected(b))
                             if b else "-"))
            else:
                b = c.get("B_descriptive")
                rows.append((f"`{t}` ({mm['row_note']})", v, _paired(c["A_descriptive"]), "descriptive only",
                             _paired(b) if b else "-", f"pays {c['pays']} of {c['n']}; descriptive only" if b else "-"))
    out.append(R.table(("Model", "Prompt", "Kappa minus the issuer lookup [95%]", "Against the issuer lookup",
                        "Payout share minus the companies' [95%]", "Payout (v1)"), rows))
    out += ["", "**Against the person** (the labelled sample; descriptive, no label). The table gives his decided "
            "labels; the second registered reading, with 'cannot tell' counted as no money back, is in the full "
            "report. " + (("The two readings fall on different sides of zero, or differ in sign, for " +
                           ", ".join(dis) + "; no conclusion is drawn there.") if dis else
                          "The two readings agree in sign and side of zero in every cell.")]
    v1 = {t: loc["models"][t]["cells"]["v1|dev"]["pays"] for t in loc["order"]}
    v2 = {t: loc["models"][t]["cells"]["v2|dev"]["pays"] for t in loc["order"]}
    h1 = [ho["models"][t]["cells"]["v1|dev"]["pays"] for t in ho["order"]]
    h2 = [ho["models"][t]["cells"]["v2|dev"]["pays"] for t in ho["order"]]
    ref = loc["order"][0]
    l1 = [v1[t] for t in loc["order"][1:]]
    l2 = [v2[t] for t in loc["order"][1:]]
    out += [
        "",
        "**Limits.**",
        "",
        "- **Not the same experiment as the local runs.** No temperature or seed could be sent, so the answers are "
        "samples at the provider's default; the provider adds its own format instructions and counts tokens its own "
        "way. Each figure is one run of each model as the provider served it on the date of the run.",
        "- **Same vendor** (the paragraph above): this arm does not measure whether that favours these models.",
        "- **No ranking.** The hosted models are not compared with one another or with the local models. The record "
        "and the one labeller agree only weakly, so a higher kappa against either is not \"better at the task\".",
        "- **Size is not isolated.** The two compared tiers differ in more than size, and the largest tier is a "
        "reference row with reasoning on. These models may have seen these complaints and their outcomes in "
        "training.",
        "",
        "**Exploratory (not registered).** Looked at after the results were seen; descriptive counts only, no test "
        f"and no interval. On DEV, judge v1 paid out on {min(l1)} to {max(l1)} of the {loc['n_dev']} complaints on the "
        f"{len(l1)} other local models and on {min(h1)} to {max(h1)} on the {len(h1)} hosted models; the study model "
        f"paid out on {v1[ref]}, the companies on {loc['record_pays_dev']} and the issuer lookup on "
        f"{loc['lookup_pays_dev']}. Judge v2 paid out on {min(l2)} to {max(l2)} on the local models and on {min(h2)} to "
        f"{max(h2)} on the hosted models (the study model: {v2[ref]}).",
        "",
        "Per cell, in the full report: seconds per call, agreement with the study model, the second reading against "
        "the person and the run record "
        f"([Hosted arm detail]({_detail(m, 'hosted')})).",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ the full report only
NOT_PRODUCED_LOCAL = (
    "accuracy against the record (registered as secondary, item 1) and against the person (item 4);",
    "judge minus TF-IDF on DEV (registered as secondary, item 2);",
    "the paired interval of the payout share minus the record's for every cell (item 3): the code computes it for v1 "
    "on DEV only, where Amendment 1, section 1 uses it for the payout label; the other cells give the count only;",
    "generation and prompt tokens per second (item 5): the code gives the wall time per call only;",
    "the list of every resend (Amendment 1, section 5) and the statement of any field left out of a request: the "
    "code lists neither. Both lists are empty here, as the run record above shows, so nothing is missing from the "
    "result; the promise was not kept in code.",
)


def b_other_local_models_detail(m):
    ext = m.get("extension")
    if not ext or not ext["local_runs"]["sends"]:
        return R.PENDING
    loc, runs = ext["local"], ext["local_runs"]
    ref = loc["order"][0]
    out = ["Added later; exploratory. Every figure below comes from `src/judge/models_analysis.py`, run unchanged "
           "(it writes `results/models/analysis.json`), or from the journal and run log of the run.", "",
           "**Run record** (from `results/models/sent.jsonl` and `results/models/run_log.jsonl`):", ""]
    out.append(R.table(("Model", "Runs", "Requests sent", "Resent", "Fields left out", "First request (UTC)",
                        "Last request (UTC)", "Status", "Other run-log events"),
                       [(f"`{t}`", r["runs"], r["sends"], r["resends"], r["fields_dropped"], _utc(r["first"]),
                         _utc(r["last"]), loc["models"][t]["status"].replace("model_finished", "finished"),
                         ", ".join(r["other_events"]) or "none") for t, r in runs["models"].items()]))
    out += ["", "**Per cell.** Malformed answers include any `done_reason` other than `stop` and any prompt over the "
            "context limit (Amendment 1, section 7). Seconds per call were measured on a laptop used for other work "
            "and are not compared between models. Agreement with the study model is kappa between the two models' "
            "decisions on the same cases (descriptive).", ""]
    rows = []
    for t in loc["order"]:
        for k, c in loc["models"][t]["cells"].items():
            s = c.get("seconds_per_call")
            rows.append((_model(t, ref), CELL_NAMES[k], f"{c['answered']} of {c['n']}", c["malformed"], c["pays"],
                         _kappa(c["kappa_vs_record"]),
                         _n(c["kappa_with_reference"]) if c.get("kappa_with_reference") is not None else "-",
                         f"{s['median']:.1f} / {s['mean']:.1f}" if s else "-"))
    out.append(R.table(("Model", "Cell", "Answered", "Malformed", "Pays", "Kappa vs record [95%]",
                        f"Kappa with `{ref}`", "Seconds per call (median / mean)"), rows))
    some_a = next(c["A"] for t in loc["order"] for c in loc["models"][t]["cells"].values() if "A" in c)
    some_b = next(c["B"] for t in loc["order"] for c in loc["models"][t]["cells"].values() if "B" in c)
    la, lb = f"{some_a['corrected_level']:.6g}", f"{some_b['corrected_level']:.6g}"
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
    out += ["", "**What the design can detect.** Amendment 1, section 2, printed with the results as the plan has it:",
            "", _plan_text("- **Source.**", "**§3 The summary sentence**"),
            "", "**Speculative decoding**, left as shipped (Amendment 1, section 4, as the plan records it; not measured "
            "per answer):", "", _plan_text("- **Speculative decoding**", "**§5 ")]
    out += ["", "**Registered outputs that the registered code does not produce.** Item numbers are those of \"What is "
            "computed\" in the registered plan. They are listed here, not computed now that the results have been seen; "
            "each is recorded as a deviation in " + F.link("docs/deviations.md", m) + ":", ""]
    out += [f"- {x}" for x in NOT_PRODUCED_LOCAL]
    return "\n".join(out)


def b_hosted_arm_detail(m):
    ext = m.get("extension")
    if not ext or not ext["hosted_runs"]["sends"]:
        return R.PENDING
    ho, runs = ext["hosted"], ext["hosted_runs"]
    ref = ho["reference"]
    out = ["Added later; exploratory. Every figure below comes from `src/judge/hosted_analysis.py`, run unchanged "
           "(it writes `results/models/hosted/analysis.json`), or from the arm's journal, run log and spend journal.",
           "", "**Run record** (from `results/models/hosted/sent.jsonl` and `results/models/hosted/run_log.jsonl`):", ""]
    out.append(R.table(("Model", "Runs", "Requests sent", "Resent", "First request (UTC)", "Last request (UTC)",
                        "Status", "Other run-log events"),
                       [(f"`{t}`", r["runs"], r["sends"], r["resends"], _utc(r["first"]), _utc(r["last"]),
                         ho["models"][t]["status"], ", ".join(r["other_events"]) or "none")
                        for t, r in runs["models"].items()]))
    out += ["", "**Per cell**, the figures the registered table does not print: seconds per call (the first call per "
            "prompt version includes compiling the schema), agreement with the study model on the same cases "
            "(descriptive), and the second registered reading against the person ('cannot tell' counted as no money "
            "back, all of the labelled sample). Where the two readings fall on different sides of zero, or have "
            "different signs, no conclusion is drawn.", ""]
    rows = []
    for t in ho["order"]:
        mm = ho["models"][t]
        for k, c in mm["cells"].items():
            s = c.get("seconds_per_call")
            p = c.get("vs_person")
            rows.append((f"`{t}`" + (" (reference row)" if not mm["compared"] else ""), CELL_NAMES[k],
                         f"{s['median']:.1f} / {s['mean']:.1f}" if s else "-", _n(c["kappa_with_reference"]),
                         _kappa(p["kappa_cannot_tell_as_no"]) if p else "-",
                         ("agree" if p["readings_agree"] else "disagree: no conclusion") if p else "-"))
    out.append(R.table(("Model", "Cell", "Seconds per call (median / mean)", f"Kappa with `{ref}`",
                        "Kappa vs the person, 'cannot tell' as no money back [95%]", "The two readings"), rows))
    out += ["", "**What the registered table and summary do not print, and where it is here.** Each is recorded as a "
            "deviation in " + F.link("docs/deviations.md", m) + ":", "",
            "- the intervals behind the labels and the corrected-level check (Amendment 2, section 10, \"printed as in "
            "Amendment 1, section 3\"), and the reference row's 95% paired intervals against the issuer lookup and "
            "the record (section 3): computed by the registered code, printed in the README section under the "
            "registered summary;",
            "- each model's status when it is \"complete\" (section 10): the table names a status only when it is "
            "not; printed in the README section's first paragraph;",
            "- the registered table writes each cell as `v1|dev`, and the `|` splits a Markdown table cell; the "
            "README escapes that one character and prints the table otherwise as the code writes it;",
            "- the code gives no accuracy, no judge minus TF-IDF, no payout interval outside v1 on DEV, and no tokens "
            "per second, as for the local arm (see " +
            f"[Other local models detail](#{DETAIL['local']})).",
            "- no resend, no lost answer and no waited-out error occurred, so the registered list of resends and "
            "charges without an answer is the sentence \"No request was resent or charged without an answer.\""]
    return "\n".join(out)
