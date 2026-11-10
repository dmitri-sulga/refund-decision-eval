"""Rules for the blind human labels that were fixed before any label existed
(docs/analysis_plan_human_labels.md)."""
from __future__ import annotations

import hashlib

AMENDMENT_1_AT = "2026-11-10T02:31:23+00:00"       # the sample and the translation source changed; still no label
PLAN_REGISTERED_AT = "2026-11-10T01:16:09+00:00"   # no label may carry an earlier timestamp


SAMPLE_SALT = "label-sample-2:"     # Amendment 1; fixed before the sample was drawn
SAMPLE_N = 100


def sample_from(train_ids: list[str], n: int = SAMPLE_N) -> list[str]:
    """Amendment 1: the labelling sample is the first n TRAIN complaints when the
    IDs are ordered by the SHA-256 of SAMPLE_SALT + ID. Nothing about a complaint
    (outcome, issuer, text, model answer) enters the draw; the order of the list
    is the order of presentation."""
    return sorted(train_ids, key=lambda i: hashlib.sha256((SAMPLE_SALT + i).encode()).hexdigest())[:n]


def halves(sample_ids: list[str]) -> dict:
    """Calibration and evaluation halves of the labelling sample: the IDs are
    ordered by the SHA-256 of the ID string; the first half is for calibration
    (a v3 prompt may be revised against it), the second half for evaluation."""
    order = sorted(sample_ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    cut = len(order) // 2
    return {"calibration": order[:cut], "evaluation": order[cut:]}


def export_calibration(annotator: str) -> str:
    """Write the calibration half only (labels, notes, system answers, recorded
    outcomes) for drafting a v3 prompt. The evaluation half is not exported."""
    from . import config as C
    from . import label as L
    from .data import load_records
    from .judgments import parse
    from .llm import Cache
    from .prompts import visible_narrative

    if not L.is_locked(annotator):
        raise SystemExit("the blind labels are not locked yet")
    labels, records, cache = L.load_labels(L.labels_path(annotator)), load_records(), Cache()
    lines = [f"# Calibration half for {annotator} (evaluation half deliberately not included)", ""]
    for i in sorted(halves(L.load_sample()["ids"])["calibration"], key=int):
        rec, lab = records[i], labels[i]
        answers = {v: parse(cache.records[Cache.key(v, i)]) for v in ("v1", "v2")}
        lines += [f"## {i}", "",
                  f"person: {lab['label']}" + (f" ({lab['note']})" if lab["note"] else ""),
                  f"recorded: {rec['company_response']}",
                  *(f"{v}: {'relief' if a['pred'] else 'no relief'} {a['fields']}" for v, a in answers.items()),
                  "", visible_narrative(rec["narrative"]), ""]
    out = C.RESULTS / "tmp" / f"{annotator}_calibration_half.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--export-calibration", metavar="ANNOTATOR", required=True)
    print(export_calibration(ap.parse_args().export_calibration))
