"""Freeze the method before the held-out run.

python -m judge.freeze            # write results/frozen.json (uses TRAIN and DEV only)

The frozen file records everything the held-out run and its report depend on:
the SHA-256 of every file that determines a prediction (settings, prompts,
client, parser, defence, baselines, metrics, runner, data), the hash of each
judge request template, the model build, the operating points chosen on DEV,
the fact-model weights and the baseline hyperparameters. `judge.run --final`
refuses to start unless all of it still matches, and the report takes the
held-out thresholds from this file, never from a later computation.
Re-freezing is allowed until the held-out run has started, never after.
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
METHOD_FILES = (
    "src/judge/config.py", "src/judge/prompts.py", "src/judge/llm.py", "src/judge/judgments.py",
    "src/judge/defence.py", "src/judge/gaming.py", "src/judge/baselines.py", "src/judge/metrics.py",
    "src/judge/run.py", "src/judge/freeze.py", "data/splits.json", "data/records.jsonl.gz",
)
ACCEPTANCE_RULE = ("v2 replaces v1 only if the 95% paired-bootstrap interval of the held-out kappa "
                   "difference (v2 minus v1) lies entirely above zero; otherwise the report says "
                   "that no improvement was demonstrated.")
_PROBE = {"product": "P", "sub_product": "S", "issue": "I", "sub_issue": "J"}


def sha256_path(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes() -> dict:
    return {f: sha256_path(C.ROOT / f) for f in METHOD_FILES}


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
    for f, h in frozen["files"].items():
        if now.get(f) != h:
            problems.append(f"{f} changed since the freeze")
    if template_hashes(frozen["final_versions"]) != frozen["request_templates"]:
        problems.append("a judge request template changed since the freeze")
    if list(heldout_ids_sha256()) != [frozen["heldout"]["n"], frozen["heldout"]["ids_sha256"]]:
        problems.append("the held-out complaint IDs changed since the freeze")
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
    return {"operating_points": ops, "fact_model": m["fact_model"], "tfidf_lr": m["tfidf_lr"]["chosen"]}


def build(m: dict) -> dict:
    return {
        "final_versions": list(FINAL_VERSIONS), "guarded_facts_version": GUARDED_FACTS_VERSION,
        "workers": WORKERS, "expected_s_per_call": m["cost"]["recent_s_per_call"],
        "environment": m["environment"],
        "request_templates": template_hashes(FINAL_VERSIONS),
        "heldout": dict(zip(("n", "ids_sha256"), heldout_ids_sha256())),
        "acceptance_rule": ACCEPTANCE_RULE,
        **snapshot(m),
        "files": file_hashes(),
    }


def main() -> None:
    from .analysis import compute
    from .run import TEST_MARKER, log_event

    if TEST_MARKER.exists():
        raise SystemExit("the held-out run has started; the method can no longer be re-frozen.")
    from .report import emit

    m = compute()
    frozen = build(m)
    FROZEN_FILE.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    log_event({"event": "method_frozen", "frozen_sha256": sha256_path(FROZEN_FILE)})
    m["frozen"] = {"sha256": sha256_path(FROZEN_FILE), "content": frozen,
                   "problems": verify_method(frozen) + verify_numbers(frozen, m)}
    problems = emit(m)
    print(f"wrote {FROZEN_FILE.relative_to(C.ROOT)} and regenerated the report; commit both before "
          "the held-out run" + ("".join("\n- " + p for p in problems)))


if __name__ == "__main__":
    main()
