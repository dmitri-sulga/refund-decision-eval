"""Rules for the blind human labels that were fixed before any label existed
(docs/analysis_plan_human_labels.md)."""
from __future__ import annotations

import hashlib

PLAN_REGISTERED_AT = "2026-11-10T01:16:09+00:00"   # no label may carry an earlier timestamp


def halves(sample_ids: list[str]) -> dict:
    """Calibration and evaluation halves of the labelling sample: the IDs are
    ordered by the SHA-256 of the ID string; the first half is for calibration
    (a v3 prompt may be revised against it), the second half for evaluation."""
    order = sorted(sample_ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    cut = len(order) // 2
    return {"calibration": order[:cut], "evaluation": order[cut:]}
