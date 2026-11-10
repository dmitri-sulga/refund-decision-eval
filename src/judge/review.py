"""Second screen, available only after the blind labels are locked.

Shows the author the v1 disagreements that were sorted individually in the
error reading (results/error_analysis/v1_dev.csv, group_basis == 'read'), with
the recorded outcome, the judge's answer and the drafted reading, and records
whether he confirms the reading or changes it. The answers go to
results/human_labels/<annotator>_error_review.jsonl and are not part of the
blind analysis.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from . import label as L
from .errors import ANNOTATION_DIR, GROUPS, load_annotations
from .judgments import parse
from .llm import Cache

GROUP_RU = {
    "model_wrong": "Ошиблась модель: неверно прочитала жалобу или неверно применила своё правило",
    "record_questionable": ("Записанное решение сомнительно или принято на усмотрение компании: она заплатила "
                            "по требованию, которое правило отклоняет, либо изменила собственный прежний ответ"),
    "undecidable": ("По тексту решить нельзя: исход зависит от того, чего в жалобе нет "
                    "(расследование банка, незавершённый спор, недостающие факты)"),
}


def review_path(annotator: str):
    return L.LABELS_DIR / f"{annotator}_error_review.jsonl"


def review_rows() -> list[dict]:
    return sorted((r for r in load_annotations("v1") if r["group_basis"] == "read"),
                  key=lambda r: int(r["complaint_id"]))


def load_reviews(path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["complaint_id"]] = r
    return out


def _labels(name: str) -> dict:
    with open(ANNOTATION_DIR / name, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = "group" if "group" in rows[0] else "category"
    return {r[key]: r["label"] for r in rows}


def next_review(annotator: str, records: dict, cache: Cache, translations) -> dict:
    if not L.is_locked(annotator):
        raise ValueError("the blind labels are not locked yet")
    rows = review_rows()
    done = load_reviews(review_path(annotator))
    todo = [r for r in rows if r["complaint_id"] not in done]
    state = {"total": len(rows), "done": len(rows) - len(todo)}
    if todo:
        r = todo[0]
        rec = records[r["complaint_id"]]
        judged = parse(cache.records[Cache.key("v1", r["complaint_id"])])
        state.update({
            "position": rows.index(r) + 1, **L.case_view(rec, translations),
            "recorded": rec["company_response"],
            "judge": {"relief": bool(judged["pred"]), "reason": judged["fields"].get("reason", "")},
            "drafted_group": r["group"], "drafted_group_en": _labels("groups.csv")[r["group"]],
            "drafted_group_ru": GROUP_RU[r["group"]],
            "category_en": _labels("categories.csv")[r["category"]], "quote": r["quote"], "note": r["note"],
            "groups": [{"group": g, "ru": GROUP_RU[g], "en": _labels("groups.csv")[g]} for g in GROUPS]})
    return state


def record_review(annotator: str, position: int, group: str, note: str) -> None:
    if not L.is_locked(annotator):
        raise ValueError("the blind labels are not locked yet")
    if group not in GROUPS:
        raise ValueError("unknown reading")
    rows = review_rows()
    path = review_path(annotator)
    todo = [r for r in rows if r["complaint_id"] not in load_reviews(path)]
    if not todo or rows.index(todo[0]) + 1 != position:
        raise ValueError("this is not the next open case")
    r = todo[0]
    L._append(path, {"complaint_id": r["complaint_id"], "group": group, "drafted_group": r["group"],
                     "confirmed": group == r["group"], "note": note.strip(), "annotator": annotator,
                     "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
