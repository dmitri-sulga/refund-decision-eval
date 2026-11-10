"""Blind labelling tool for one person (terminal, standard library only, no network).

Shows one complaint at a time with exactly the text the models see (product,
issue, narrative) and nothing else: no recorded outcome, no model output, no
company field, no date, no complaint ID. Records money back / no money back /
cannot tell and an optional note to results/human_labels/<annotator>.jsonl.
Resumable: answered cases are skipped on the next start.

python -m judge.label --annotator owner
python -m judge.label --make-sample        # recreate data/label_sample.json (deterministic)

The rubric is in docs/labelling_rubric.md. This module deliberately imports
neither the response cache nor the analysis code.
"""
from __future__ import annotations

import argparse
import json
import shutil
import textwrap
import time
from datetime import datetime, timezone

import numpy as np

from . import config as C
from .data import load_records, load_splits
from .prompts import _case_header, visible_narrative

SAMPLE_FILE = C.DATA / "label_sample.json"
LABELS_DIR = C.RESULTS / "human_labels"
SAMPLE_SEED = 11
SAMPLE_N = 100
CHOICES = {"1": "relief", "2": "no_relief", "3": "cannot_tell"}
PROMPT = "[1] money back   [2] no money back   [3] cannot tell   [u] undo previous   [q] quit > "


def make_sample(split: str = "dev", n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> dict:
    """A seeded random draw of complaint IDs in presentation order. The recorded
    outcomes are not read, so the sample is not stratified by them."""
    if split == "test" and not _heldout_finished():
        raise SystemExit("held-out complaints may be shown to a person only after the held-out run.")
    ids = sorted(load_splits()[split], key=int)
    order = np.random.default_rng(seed).permutation(len(ids))[:n]
    return {"split": split, "seed": seed, "n": int(min(n, len(ids))), "ids": [ids[k] for k in order]}


def _heldout_finished() -> bool:
    marker = C.RESULTS / "test_run.json"
    return marker.exists() and bool(json.loads(marker.read_text()).get("finished_at"))


def load_sample() -> dict:
    return json.loads(SAMPLE_FILE.read_text())


def load_labels(path) -> dict:
    """{complaint_id: record}; the last record of a complaint wins, an undo removes it."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("undo"):
                    out.pop(r["complaint_id"], None)
                else:
                    out[r["complaint_id"]] = r
    return out


def render(rec: dict, position: int, total: int, width: int = 100) -> str:
    """Everything the annotator is shown for one complaint."""
    body = []
    for para in visible_narrative(rec["narrative"]).split("\n"):
        body += textwrap.wrap(para, width) or [""]
    return "\n".join(["=" * width, f"Case {position} of {total}", "-" * width,
                      _case_header(rec).rstrip(), "-" * width, *body, "=" * width])


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def session(annotator: str, ask=input, show=print, clock=time.monotonic) -> int:
    """Run until the sample is done or the annotator quits; returns cases answered now."""
    sample = load_sample()
    if sample["split"] == "test" and not _heldout_finished():
        raise SystemExit("held-out complaints may be shown to a person only after the held-out run.")
    records = load_records()
    path = LABELS_DIR / f"{annotator}.jsonl"
    width = min(110, shutil.get_terminal_size((100, 40)).columns)
    answered, last = 0, None
    while True:
        done = load_labels(path)
        todo = [i for i in sample["ids"] if i not in done]
        if not todo:
            show(f"All {len(sample['ids'])} cases are labelled. Thank you.")
            return answered
        cid = todo[0]
        show(render(records[cid], sample["ids"].index(cid) + 1, len(sample["ids"]), width))
        t0 = clock()
        while True:
            a = ask(PROMPT).strip().lower()
            if a in CHOICES or a == "q" or (a == "u" and last):
                break
        if a == "q":
            show(f"Saved. {len(done)} of {len(sample['ids'])} labelled; run the same command to continue.")
            return answered
        if a == "u":
            _append(path, {"complaint_id": last, "undo": True, "annotator": annotator})
            last = None
            continue
        seconds = round(clock() - t0, 1)
        note = ask("note (optional, Enter to skip) > ").strip()
        _append(path, {"complaint_id": cid, "label": CHOICES[a], "note": note, "annotator": annotator,
                       "seconds": seconds,
                       "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        answered, last = answered + 1, cid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="owner")
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    a = ap.parse_args()
    if a.make_sample:
        SAMPLE_FILE.write_text(json.dumps(make_sample(a.split), indent=1) + "\n")
        print(f"wrote {SAMPLE_FILE.relative_to(C.ROOT)}")
        return
    session(a.annotator)


if __name__ == "__main__":
    main()
