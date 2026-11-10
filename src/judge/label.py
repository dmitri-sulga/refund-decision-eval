"""Blind labelling tool for one person (local only: a page served on 127.0.0.1, or the terminal).

Shows one complaint at a time with exactly the text the models see (product,
issue, narrative) and nothing else: no recorded outcome, no model output, no
company field, no date, no complaint ID, no error category. Beside the English
original it shows a Russian machine translation (translate.py) made by a
separate model, not the judged one: a reading aid that no judged system reads. It records money back / no money back /
cannot tell, an optional note and the time per case to
results/human_labels/<annotator>.jsonl. Resumable: answered cases are skipped on
the next start. The answers are locked when the last case is answered.

python -m judge.label                      # opens the labelling page in the browser
python -m judge.label --terminal           # the same in the terminal
python -m judge.label --make-sample        # recreate data/label_sample.json (deterministic)

The rubric is in docs/labelling_rubric.md (Russian: docs/labelling_rubric_ru.md).
This module deliberately imports neither the response cache nor the analysis
code; the screen shown after locking lives in review.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
import time
from datetime import datetime, timezone

from . import config as C
from .data import load_records, load_splits
from .prompts import _case_header, visible_narrative
from .translate import Translations, problems

SAMPLE_FILE = C.DATA / "label_sample.json"
LABELS_DIR = C.RESULTS / "human_labels"
CHOICES = {"1": "relief", "2": "no_relief", "3": "cannot_tell"}
CHOICE_TEXT = {"relief": ("Вернуть деньги", "money back"),
               "no_relief": ("Не возвращать деньги", "no money back"),
               "cannot_tell": ("Не могу решить по тексту", "cannot tell")}
PROMPT = ("[1] вернуть деньги / money back   [2] не возвращать / no money back   "
          "[3] не могу решить / cannot tell   [u] отменить предыдущий / undo   [q] выйти / quit > ")
AID_NOTE = ("Русский текст — машинный перевод, сделанный отдельной моделью (не той, которую проверяют), "
            "только для удобства чтения; он может содержать ошибки. Оригинал — английский текст.")
# header terms, translated by the AI coding assistant that wrote this code (the narratives are
# machine-translated by a separate model, see translate.py)
HEADER_RU = {
    "Credit card": "Кредитная карта",
    "General-purpose credit card or charge card": "Кредитная или расчётная (charge) карта общего назначения",
    "Store credit card": "Кредитная карта магазина",
    "Problem with a purchase shown on your statement": "Проблема с покупкой, указанной в выписке",
    "Card was charged for something you did not purchase with the card":
        "С карты списали деньги за то, что вы этой картой не покупали",
    "Overcharged for something you did purchase with the card":
        "Списали больше, чем нужно, за покупку, которую вы действительно сделали этой картой",
    "Credit card company isn't resolving a dispute about a purchase on your statement":
        "Эмитент карты не решает спор о покупке, указанной в выписке",
}


# ------------------------------------------------------------------ sample
def make_sample() -> dict:
    """The labelling sample (Amendment 1 of docs/analysis_plan_human_labels.md):
    the first 100 TRAIN complaints by a salted hash of the ID, in that order.
    Recorded outcomes, issuers, texts and model answers do not enter the draw."""
    from .human import SAMPLE_N, SAMPLE_SALT, sample_from

    ids = sample_from(load_splits()["train"], SAMPLE_N)
    return {"split": "train", "rule": f"first {SAMPLE_N} TRAIN IDs ordered by sha256('{SAMPLE_SALT}' + id)",
            "n": len(ids), "ids": ids}


def load_sample() -> dict:
    return json.loads(SAMPLE_FILE.read_text())


# ------------------------------------------------------------------ storage
def labels_path(annotator: str):
    return LABELS_DIR / f"{annotator}.jsonl"


def lock_path(annotator: str):
    return LABELS_DIR / f"{annotator}.lock.json"


def load_labels(path) -> dict:
    """{complaint_id: record}; the last record of a complaint wins, an undo removes it."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.pop(r["complaint_id"], None)      # re-inserting keeps the file order
                if not r.get("undo"):
                    out[r["complaint_id"]] = r
    return out


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def is_locked(annotator: str) -> bool:
    """True once the blind pass is complete and sealed. A labels file that was
    changed after locking is an error, not an unlocked state."""
    lp = lock_path(annotator)
    if not lp.exists():
        return False
    lock = json.loads(lp.read_text())
    if hashlib.sha256(labels_path(annotator).read_bytes()).hexdigest() != lock["labels_sha256"]:
        raise SystemExit(f"{labels_path(annotator)} changed after it was locked.")
    return True


def _lock(annotator: str) -> None:
    path = labels_path(annotator)
    lock_path(annotator).write_text(json.dumps({
        "annotator": annotator, "n": len(load_labels(path)),
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=1) + "\n")


# ------------------------------------------------------------------ what is shown
def case_view(rec: dict, translations: Translations | None) -> dict:
    """Everything the annotator is shown for one complaint (no ID, no outcome)."""
    text = visible_narrative(rec["narrative"])
    tr = translations.get(text, rec["complaint_id"]) if translations else None
    return {"header_en": _case_header(rec).rstrip().split("\n"),
            "header_ru": [f"Продукт: {HEADER_RU.get(rec['product'], rec['product'])} / "
                          f"{HEADER_RU.get(rec['sub_product'], rec['sub_product'])}",
                          f"Проблема, выбранная потребителем: {HEADER_RU.get(rec['issue'], rec['issue'])} / "
                          f"{HEADER_RU.get(rec['sub_issue'], rec['sub_issue'])}"],
            "narrative_en": text, "narrative_ru": tr["content"] if tr else None,
            "translation_problems": problems(text, tr) if tr else ["нет перевода / no translation"]}


def next_case(annotator: str, records: dict, translations: Translations | None) -> dict:
    sample = load_sample()
    if sample["split"] != "train":
        raise SystemExit("the labelling sample must come from TRAIN (Amendment 1 of the plan).")
    locked = is_locked(annotator)
    done = load_labels(labels_path(annotator))
    todo = [i for i in sample["ids"] if i not in done]
    state = {"total": len(sample["ids"]), "done": len(sample["ids"]) - len(todo), "locked": locked,
             "can_undo": bool(done) and not locked}
    if todo and not locked:
        state.update({"position": sample["ids"].index(todo[0]) + 1, **case_view(records[todo[0]], translations)})
    return state


def record_answer(annotator: str, position: int, label: str, note: str, seconds: float,
                  ui: str, view: dict) -> None:
    """Store the answer to the case at `position` (which must be the next open one)."""
    if is_locked(annotator):
        raise ValueError("the answers are locked")
    if label not in CHOICE_TEXT:
        raise ValueError("unknown answer")
    sample = load_sample()
    path = labels_path(annotator)
    todo = [i for i in sample["ids"] if i not in load_labels(path)]
    if not todo or sample["ids"].index(todo[0]) + 1 != position:
        raise ValueError("this is not the next open case")
    _append(path, {"complaint_id": todo[0], "label": label, "note": note.strip(), "annotator": annotator,
                   "seconds": round(float(seconds), 1), "ui": ui,
                   "translation_shown": view["narrative_ru"] is not None,
                   "translation_problems": view["translation_problems"] if view["narrative_ru"] else [],
                   "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if len(todo) == 1:            # that was the last case: seal the blind pass
        _lock(annotator)


def undo_last(annotator: str) -> bool:
    """Remove the most recent answer (not possible once locked)."""
    if is_locked(annotator):
        raise ValueError("the answers are locked")
    path = labels_path(annotator)
    done = load_labels(path)
    if not done:
        return False
    last = list(done)[-1]          # load_labels keeps the order in which the answers were written
    _append(path, {"complaint_id": last, "undo": True, "annotator": annotator})
    return True


# ------------------------------------------------------------------ terminal
def render(rec: dict, position: int, total: int, width: int = 100,
           translations: Translations | None = None) -> str:
    """The terminal screen for one complaint: Russian aid first, English original below."""
    v = case_view(rec, translations)

    def wrap(text):
        out = []
        for para in text.split("\n"):
            out += textwrap.wrap(para, width) or [""]
        return out

    lines = ["=" * width, f"Случай {position} из {total} / Case {position} of {total}", "-" * width]
    if v["narrative_ru"]:
        lines += [AID_NOTE] + (["ВНИМАНИЕ: перевод может быть неполным: " + ", ".join(v["translation_problems"])]
                               if v["translation_problems"] else [])
        lines += ["", *v["header_ru"], "", *wrap(v["narrative_ru"]), "-" * width, "ORIGINAL (English):"]
    lines += [*v["header_en"], "", *wrap(v["narrative_en"]), "=" * width]
    return "\n".join(lines)


def session(annotator: str, ask=input, show=print, clock=time.monotonic, translations=None) -> int:
    """Terminal labelling until the sample is done or the annotator quits; returns cases answered now."""
    records = load_records()
    width = min(110, shutil.get_terminal_size((100, 40)).columns)
    answered = 0
    while True:
        state = next_case(annotator, records, translations)
        if "position" not in state:
            show(f"Все {state['total']} случаев размечены, ответы зафиксированы. Спасибо. / "
                 f"All {state['total']} cases are labelled and locked. Thank you.")
            return answered
        cid = load_sample()["ids"][state["position"] - 1]
        show(render(records[cid], state["position"], state["total"], width, translations))
        t0 = clock()
        while True:
            a = ask(PROMPT).strip().lower()
            if a in CHOICES or a == "q" or (a == "u" and state["can_undo"]):
                break
        if a == "q":
            show(f"Сохранено: {state['done']} из {state['total']}. Запустите ту же команду, чтобы продолжить. / "
                 f"Saved. {state['done']} of {state['total']} labelled; run the same command to continue.")
            return answered
        if a == "u":
            undo_last(annotator)
            continue
        seconds = clock() - t0
        note = ask("заметка (необязательно, Enter — пропустить) / note (optional) > ")
        record_answer(annotator, state["position"], CHOICES[a], note, seconds, "terminal", state)
        answered += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="owner")
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--terminal", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.make_sample:
        SAMPLE_FILE.write_text(json.dumps(make_sample(), indent=1) + "\n")
        print(f"wrote {SAMPLE_FILE.relative_to(C.ROOT)}")
        return
    if a.terminal:
        session(a.annotator, translations=Translations())
        return
    from .labelserver import serve
    serve(a.annotator, a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
