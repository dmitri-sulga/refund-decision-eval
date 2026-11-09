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
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


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
        ("Gaming subset", f"{s['n_gaming']} DEV complaints x {s['n_attacks']} rewrites, seed {s['gaming_seed']}"),
        ("Near-duplicate rule", f"word-trigram TF-IDF cosine >= {s['near_dup_cosine']} (collapsed); "
                                f"ceiling proxy uses same-company pairs >= {s['ceiling_pair_cosine']}"),
    ]
    return table(("Setting", "Value"), rows)


def b_data(m):
    d = m["data"]
    f = d["funnel"]
    rows = [
        ("Rows in the archived official bulk export", f"{d['mirror_rows_total']:,}"),
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


BLOCKS = {
    "settings": b_settings,
    "data": b_data,
    "base_rates": b_base_rates,
}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    check = ap.parse_args().check
    m = compute()
    metrics_text = json.dumps(m, indent=1, sort_keys=True, default=float) + "\n"
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
    bad = number_guard(readme_new)
    if bad:
        problems.append("README has result-like numbers outside generated blocks: " + ", ".join(bad))
    if problems:
        print("\n\n".join(problems))
        sys.exit(1)
    print("check passed: README, metrics.json and report.md match the cached results"
          if check else "regenerated README.md, results/metrics.json, results/report.md")


if __name__ == "__main__":
    main()
