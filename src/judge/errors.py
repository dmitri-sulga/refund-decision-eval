"""Error analysis of a judge version on DEV.

python -m judge.errors --export v1
    writes results/tmp/v1_dev_disagreements.md (every disagreement, full text) for reading.

The annotations live in results/error_analysis/<version>_dev.csv, one row per
disagreement: complaint_id, category, group, group_basis, quote, note. The group
is one of three coarse readings (model wrong / recorded decision questionable /
not decidable from the text); group_basis says whether it was assigned by reading
the case ('read') or follows from the definition of its category ('category'). `validate` checks that every
disagreement is annotated exactly once (no cherry-picking) and that every quote
is a verbatim substring of the complaint narrative (no invented evidence).
"""
from __future__ import annotations

import argparse
import csv
import re

from . import config as C
from .data import load_records, load_splits
from .judgments import load

ANNOTATION_DIR = C.RESULTS / "error_analysis"
GROUPS = ("model_wrong", "record_questionable", "undecidable")


def disagreements(version: str, records=None, splits=None) -> list[dict]:
    records = records or load_records()
    splits = splits or load_splits()
    preds = load(version, splits["dev"])
    out = []
    for cid in splits["dev"]:
        p = preds[cid]
        if p["pred"] != records[cid]["label"]:
            out.append({"complaint_id": cid, "label": records[cid]["label"], "pred": p["pred"],
                        "p_token": p["p_token"], "fields": p["fields"], "valid": p["valid"]})
    return out


def direction(label: int, pred: int) -> str:
    return "judge relief, company none" if pred == 1 else "judge none, company relief"


def export(version: str) -> str:
    records = load_records()
    rows = disagreements(version, records)
    lines = [f"# {version} DEV disagreements ({len(rows)})", ""]
    for r in rows:
        rec = records[r["complaint_id"]]
        lines += [f"## {r['complaint_id']} | {direction(r['label'], r['pred'])} | "
                  f"p_token={r['p_token']:.3f} | {rec['company']} | {rec['sub_issue']}",
                  "", "judge: " + str(r["fields"]), "", rec["narrative"], ""]
    out = C.RESULTS / "tmp" / f"{version}_dev_disagreements.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return str(out)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_annotations(version: str) -> list[dict]:
    path = ANNOTATION_DIR / f"{version}_dev.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(version: str, rows: list[dict], records: dict, expected_ids: set) -> None:
    ids = [r["complaint_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("a disagreement is annotated twice")
    if set(ids) != expected_ids:
        missing, extra = expected_ids - set(ids), set(ids) - expected_ids
        raise ValueError(f"annotations do not match disagreements: missing {sorted(missing)[:5]}, "
                         f"extra {sorted(extra)[:5]}")
    for r in rows:
        if r["quote"] and _norm(r["quote"]) not in _norm(records[r["complaint_id"]]["narrative"]):
            raise ValueError(f"quote for {r['complaint_id']} is not in the narrative")
        if r["group"] not in GROUPS or r["group_basis"] not in ("read", "category"):
            raise ValueError(f"bad group for {r['complaint_id']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    print(export(ap.parse_args().export))
