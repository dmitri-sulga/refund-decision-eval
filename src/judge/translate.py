"""Russian translations of the labelling sample: a reading aid only.

The person who labels the sample reads English at an intermediate level, so the
labelling tool shows a Russian translation beside the original English text.
The translations were produced by a separate model, not the judged one, and
imported here as plain text; they may contain errors, and the English original
is what counts. No judge, baseline or metric reads this file.

python -m judge.translate --import-dir DIR --index FILE   # build results/cache/translations.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .cues import amounts
from .data import load_records
from .prompts import visible_narrative

TRANSLATION_FILE = C.RESULTS / "cache" / "translations.jsonl"
TRANSLATOR = "a separate model, not the judged one"


def source_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def problems(source: str, rec: dict) -> list[str]:
    """Cheap checks that a translation is complete enough to read; any problem is
    shown to the annotator as a warning to rely on the English original."""
    out = []
    text = rec["content"]
    letters = [c for c in text if c.isalpha() and c not in "Xx"]
    if not letters or sum("Ѐ" <= c <= "ӿ" for c in letters) / len(letters) < 0.6:
        out.append("not mostly Russian")
    if not 0.5 <= len(text) / max(1, len(source)) <= 2.5:
        out.append("length differs a lot from the original")
    if sorted(amounts(source)) != sorted(amounts(text)):
        out.append("dollar amounts differ from the original")
    return out


class Translations:
    def __init__(self, path=TRANSLATION_FILE):
        self.records = {}
        if Path(path).exists():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[r["complaint_id"]] = r

    def get(self, text: str, complaint_id: str) -> dict | None:
        """The stored translation of exactly this text, or None."""
        rec = self.records.get(complaint_id)
        return rec if rec is not None and rec["source_sha256"] == source_sha(text) else None


def import_dir(directory: str, index_file: str) -> dict:
    """Build the translation file from one text file per complaint (item_<n>.txt)
    and the index that maps n to a complaint ID."""
    records = load_records()
    index = json.loads(Path(index_file).read_text())
    rows, missing, flagged = [], [], []
    for n, cid in sorted(index.items(), key=lambda kv: int(kv[0])):
        path = Path(directory) / f"item_{n}.txt"
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            missing.append(cid)
            continue
        source = visible_narrative(records[cid]["narrative"])
        rec = {"complaint_id": cid, "source_sha256": source_sha(source), "translator": TRANSLATOR,
               "content": path.read_text(encoding="utf-8").strip(),
               "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if problems(source, rec):
            flagged.append((cid, problems(source, rec)))
        rows.append(rec)
    TRANSLATION_FILE.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                                encoding="utf-8")
    return {"written": len(rows), "missing": missing, "flagged": flagged}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-dir", required=True)
    ap.add_argument("--index", required=True)
    a = ap.parse_args()
    print(json.dumps(import_dir(a.import_dir, a.index), ensure_ascii=False, indent=1))
