"""Freeze the method before the held-out run.

python -m judge.freeze            # write results/frozen.json (uses TRAIN and DEV only)

The frozen file records what the held-out run and its report depend on: the
SHA-256 of every source file, of the pinned requirements, of the data, plan,
rubric and annotation files; the hash of each judge request template; the model
build; an append-only fingerprint of the answer caches; the operating points
chosen on DEV, the fact-model weights, the baseline hyperparameters and the
chosen keyword rule; and every claim that will be made about the held-out
split, each with its decision rule. `judge.run --final` refuses to start unless
all of it still matches and the working tree is committed, and the report takes
the held-out thresholds and rules from this file, never from a later
computation. Re-freezing is allowed until the held-out run has started, never
after. What a local file cannot give is a timestamp the author does not
control: the commit that contains it has to be published before the run.
"""
from __future__ import annotations

import hashlib
import json
import math

from . import config as C

FROZEN_FILE = C.RESULTS / "frozen.json"
FINAL_VERSIONS = ("v1", "v2")        # judge versions sent to the model in the held-out run
GUARDED_FACTS_VERSION = "v2"         # the guarded pipeline reads this version's extracted facts
WORKERS = 1                          # sequential calls (see the concurrency check)
METHOD_GLOBS = ("src/judge/*.py", "requirements.txt", "Makefile", "data/splits.json", "data/records.jsonl.gz",
                "data/label_sample.json", "data/near_duplicate_pairs.json", "docs/*.md",
                "results/error_analysis/*", "results/baseline_cost.json")
CACHES = ("results/cache/responses.jsonl", "results/cache/replays.jsonl")   # append-only: a prefix is fingerprinted
_PROBE = {"product": "P", "sub_product": "S", "issue": "I", "sub_issue": "J"}

# Every statement the report will make about the held-out split, with the rule that
# decides it. "Interval" is the 95% percentile bootstrap interval over complaints
# (paired where two systems are compared); "Wilson" the 95% Wilson score interval.
CLAIMS = [
    {"id": "v2_vs_v1",
     "rule": "Kappa of v2 minus kappa of v1, paired interval. Entirely above zero: 'v2 replaces v1'. "
             "Otherwise: 'no improvement of v2 over v1 was demonstrated'."},
    {"id": "judge_vs_baseline",
     "rule": "For each judge version (v1, v2) and each of the TF-IDF model, the issuer relief rate and the "
             "company rule: kappa of the judge minus kappa of the baseline, paired interval. Entirely below "
             "zero: 'the judge is below the baseline'. Entirely above: 'the judge is above the baseline'. "
             "Otherwise: 'not distinguishable'."},
    {"id": "within_issuer",
     "rule": "Within-issuer AUC of TF-IDF minus that of each judge version, interval. Entirely above zero: "
             "'TF-IDF orders complaints better than the judge within the same issuer'. Entirely below: the "
             "reverse. Otherwise: 'the lead of TF-IDF is not demonstrated once the issuer is held fixed'."},
    {"id": "tfidf_operating_point",
     "rule": "The TF-IDF threshold frozen on DEV, applied unchanged. Fewer automated held-out cases than the "
             "minimum in the settings: 'too few automated cases to judge'. Otherwise the Wilson interval of "
             "agreement on the automated cases: lower bound at or above the target: 'meets the target'; upper "
             "bound below it: 'misses the target'; else 'not shown to meet the target'. If agreement is not "
             "above that of always answering no relief on the same cases, the report adds that the automation "
             "does no better than that rule."},
    {"id": "llm_automation",
     "rule": "No threshold reached the target on DEV for judge v1, judge v2 or the guarded pipeline, so "
             "none automates a held-out case: 'the LLM judge automates nothing'. Their best threshold chosen "
             "on the held-out split itself is shown as an optimistic bound only."},
    {"id": "noise_pass",
     "rule": "After the scored pass, every held-out request is sent a second time and stored apart "
             "(results/cache/replays.jsonl). The second answers are never scored against recorded "
             "decisions; the report gives only how often the decision repeated, the kappa between the two "
             "passes and the shifts in P(relief)."},
]


def sha256_path(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_files() -> list[str]:
    out = set()
    for pattern in METHOD_GLOBS:
        out |= {p.relative_to(C.ROOT).as_posix() for p in C.ROOT.glob(pattern) if p.is_file()}
    return sorted(out)


def file_hashes() -> dict:
    return {f: sha256_path(C.ROOT / f) for f in method_files()}


def cache_fingerprints() -> dict:
    out = {}
    for f in CACHES:
        data = (C.ROOT / f).read_bytes()
        out[f] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return out


def template_hashes(versions) -> dict:
    """Hash of the full request (model, options, system prompt, user template,
    schema) of each version for a fixed dummy complaint."""
    from . import llm
    from .prompts import build_messages

    out = {}
    for v in versions:
        messages, schema = build_messages(v, _PROBE, "narrative")
        out[v] = llm.request_hash(llm.request_body(messages, schema))
    return out


def heldout_ids_sha256() -> tuple[int, str]:
    """Number of held-out complaint IDs and the SHA-256 of them, sorted and joined by newlines."""
    ids = sorted(json.loads((C.DATA / "splits.json").read_text())["ids"]["test"], key=int)
    return len(ids), hashlib.sha256("\n".join(ids).encode()).hexdigest()


def load() -> dict | None:
    return json.loads(FROZEN_FILE.read_text()) if FROZEN_FILE.exists() else None


def verify_method(frozen: dict) -> list[str]:
    """Problems that make the frozen method differ from the code and data on disk."""
    problems = []
    now = file_hashes()
    for f in sorted(set(now) | set(frozen["files"])):
        if now.get(f) != frozen["files"].get(f):
            problems.append(f"{f} changed since the freeze")
    for f, fp in frozen["caches"].items():
        data = (C.ROOT / f).read_bytes()
        if len(data) < fp["bytes"] or hashlib.sha256(data[: fp["bytes"]]).hexdigest() != fp["sha256"]:
            problems.append(f"{f}: the answers that existed at the freeze were changed")
    if template_hashes(frozen["final_versions"]) != frozen["request_templates"]:
        problems.append("a judge request template changed since the freeze")
    if list(heldout_ids_sha256()) != [frozen["heldout"]["n"], frozen["heldout"]["ids_sha256"]]:
        problems.append("the held-out complaint IDs changed since the freeze")
    if frozen["claims"] != CLAIMS:
        problems.append("the held-out claims or their rules changed since the freeze")
    return problems


def _close(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, abs_tol=tol)


def verify_numbers(frozen: dict, m: dict) -> list[str]:
    """Problems that make the frozen DEV choices differ from a fresh computation."""
    problems = []
    fresh = snapshot(m)
    for name, op in frozen["operating_points"].items():
        if not _close(op["threshold"], fresh["operating_points"].get(name, {}).get("threshold")):
            problems.append(f"operating point of {name} differs from the frozen one")
    if set(fresh["operating_points"]) != set(frozen["operating_points"]):
        problems.append("the set of systems with an operating point changed")
    fz, fr = frozen["fact_model"], fresh["fact_model"]
    if fz["vocab"] != fr["vocab"] or not all(_close(a, b) for a, b in
                                             zip(fz["coef"] + [fz["intercept"]], fr["coef"] + [fr["intercept"]])):
        problems.append("fact model differs from the frozen one")
    if frozen["tfidf_lr"] != fresh["tfidf_lr"]:
        problems.append("baseline hyperparameters differ from the frozen ones")
    if frozen["keyword_rule"] != fresh["keyword_rule"]:
        problems.append("the chosen keyword rule differs from the frozen one")
    return problems


def snapshot(m: dict) -> dict:
    """The DEV-side choices that the held-out report will use, from computed metrics."""
    ops = {}
    for name, sigs in m["automation_dev"].items():
        r = sigs["token"]
        at = r["at_threshold"]
        ops[name] = {"signal": "token", "threshold": r["threshold"],
                     "dev_automated": at["n_auto"] if at else 0,
                     "dev_agreement_on_automated": at["agreement"]["value"] if at else None}
    return {"operating_points": ops, "fact_model": m["fact_model"], "tfidf_lr": m["tfidf_lr"]["chosen"],
            "keyword_rule": {"rule": m["keyword_rule"]["chosen"]["rule"], "cues": m["keyword_rule"]["chosen"]["cues"]}}


def build(m: dict) -> dict:
    return {
        "final_versions": list(FINAL_VERSIONS), "guarded_facts_version": GUARDED_FACTS_VERSION,
        "workers": WORKERS, "expected_s_per_call": m["cost"]["recent_s_per_call"],
        "environment": m["environment"],
        "request_templates": template_hashes(FINAL_VERSIONS),
        "heldout": dict(zip(("n", "ids_sha256"), heldout_ids_sha256())),
        "claims": CLAIMS,
        "settings": {"target_agreement": C.TARGET_AGREEMENT, "min_automated": C.MIN_AUTOMATED,
                     "bootstrap_b": C.BOOTSTRAP_B, "bootstrap_seed": C.BOOTSTRAP_SEED},
        **snapshot(m),
        "caches": cache_fingerprints(),
        "files": file_hashes(),
    }


def main() -> None:
    from .analysis import compute
    from .report import emit
    from .run import TEST_MARKER, log_event

    if TEST_MARKER.exists():
        raise SystemExit("the held-out run has started; the method can no longer be re-frozen.")
    m = compute()
    log_event({"event": "method_frozen"})          # logged first: the run log is part of what the report reads
    frozen = build(m)
    FROZEN_FILE.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    frozen = load()                               # exactly what a later report will read
    m["frozen"] = {"sha256": sha256_path(FROZEN_FILE), "content": frozen,
                   "problems": verify_method(frozen) + verify_numbers(frozen, m)}
    problems = emit(m)
    print(f"wrote {FROZEN_FILE.relative_to(C.ROOT)} (SHA-256 {sha256_path(FROZEN_FILE)}) and regenerated the "
          "report; commit both, and publish the commit hash, before the held-out run"
          + ("".join("\n- " + p for p in problems)))


if __name__ == "__main__":
    main()
