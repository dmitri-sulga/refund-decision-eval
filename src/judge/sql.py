"""Some of the study's figures again, in SQL. Added after the held-out run and after the blind labels were
locked; it decides nothing and feeds no reported number (results/post_run_changes.json records it). Most of what
the queries print was already reported; the rest is a descriptive breakdown printed for the first time (each
issuer's TRAIN count, paid count, raw and smoothed rate, share of TRAIN and running share; the same-issuer pairs
per issuer), added later and descriptive (docs/sql.md says which is which).

python -m judge.sql                  # load the committed data into an in-memory SQLite database, run sql/*.sql
python -m judge.sql --rows 0         # print every row
python -m judge.sql --db /tmp/j.db   # also keep the database in a new file outside the repository, to explore it
python -m judge.sql --docs           # rewrite the expected-output blocks of docs/sql.md from a fresh run

Two tables, loaded with the standard library's sqlite3 only:

- complaints: every complaint of data/records.jsonl.gz that a split uses, with its split (data/splits.json), its
  issuer (the company field exactly as the study uses it: the study does not normalise issuer names), the recorded
  company response, the label the study derives from it (checked against data.label_of, the study's own mapping)
  and the date the CFPB received it, the only date field in the records.
- systems: the system names and order of the README table, and whether the system gives a ranking score
  (SCORED in analysis.py).
- decisions: the decision of every system on every held-out complaint, from the same code as the report: the
  judges and the guarded pipeline from the answer caches, the classic baselines refitted on TRAIN by the frozen
  code, reusing the hyperparameters recorded at the freeze (they have no cached answers; the report reruns the
  cross-validated search on TRAIN, which picks the same values). `score` is the score the frozen ranking uses
  (NULL where there is none, or where the model gave no usable probability).

Nothing is written into the repository unless --docs is given, which rewrites the expected-output blocks of
docs/sql.md; the database lives in memory unless --db names a new file outside the repository.
"""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from pathlib import Path

from . import config as C

SQL_DIR = C.ROOT / "sql"
DOC = C.ROOT / "docs" / "sql.md"
ROWS = 16      # rows printed per result unless --rows says otherwise
DOC_BLOCK_RE = re.compile(r"(<!-- BEGIN output:([\w.]+) -->\n)(.*?)(<!-- END output:\2 -->)", re.S)

SCHEMA = """
CREATE TABLE complaints (
    complaint_id     TEXT PRIMARY KEY,
    split            TEXT NOT NULL CHECK (split IN ('train', 'dev', 'test', 'debug')),
    issuer           TEXT NOT NULL,
    company_response TEXT NOT NULL,
    label            INTEGER NOT NULL CHECK (label IN (0, 1)),   -- 1 = the company recorded monetary relief
    date_received    TEXT NOT NULL                               -- ISO date, yyyy-mm-dd
);
CREATE TABLE systems (
    system       TEXT PRIMARY KEY,
    position     INTEGER NOT NULL,    -- the order of the README table
    display_name TEXT NOT NULL,
    scored       INTEGER NOT NULL CHECK (scored IN (0, 1))   -- 1 = the system gives a ranking score
);
CREATE TABLE decisions (
    system       TEXT NOT NULL REFERENCES systems (system),
    complaint_id TEXT NOT NULL REFERENCES complaints (complaint_id),
    decision     INTEGER NOT NULL CHECK (decision IN (0, 1)),   -- 1 = the system pays
    score        REAL,                                          -- the ranking score, NULL if none
    PRIMARY KEY (system, complaint_id)
);
"""


def complaint_rows(records: dict, splits: dict) -> list[tuple]:
    from .data import label_of

    rows = []
    for split, ids in splits.items():
        for i in ids:
            r = records[i]
            if label_of(r["company_response"]) != r["label"]:
                raise RuntimeError(f"complaint {i}: label does not follow the study's mapping")
            rows.append((i, split, r["company"], r["company_response"], r["label"], r["date_received"]))
    return rows


def systems(records: dict, splits: dict):
    """The report's systems, with the TF-IDF hyperparameters recorded at the freeze (no new search; the report
    reruns the search on TRAIN, which picks the same values)."""
    from . import freeze
    from .analysis import Systems
    from .llm import Cache

    return Systems(records, splits, Cache(), lr_params=freeze.load()["tfidf_lr"])


def decision_rows(S, split: str = "test") -> list[tuple]:
    """Every system's decision and ranking score on `split`, from the frozen code and the caches."""
    from .analysis import SCORED, SYSTEMS

    splits = S.splits
    rows = []
    for name in SYSTEMS:
        pr = S.predict(name, split)
        if pr is None:
            raise RuntimeError(f"no cached answers for {name} on {split}")
        score = pr.get("p_raw", pr["p"]) if name in SCORED else [math.nan] * len(splits[split])
        for i, d, s in zip(splits[split], pr["pred"], score):
            rows.append((name, i, int(d), None if s != s else float(s)))
    return rows


def connect(path: str | Path = ":memory:", S=None) -> sqlite3.Connection:
    """A database with the tables filled from the committed data and caches (S: a Systems object to reuse)."""
    from .analysis import SCORED
    from .data import load_records, load_splits
    from .frontpage import SHORT

    if str(path) != ":memory:":
        p = Path(path).resolve()
        if p == C.ROOT or C.ROOT in p.parents:
            raise SystemExit(f"refusing to write a database inside the repository: {p}")
        if p.exists():
            raise SystemExit(f"{p} exists; name a new file (nothing is overwritten)")
    if S is None:
        S = systems(load_records(), load_splits())
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?)", complaint_rows(S.records, S.splits))
    con.executemany("INSERT INTO systems VALUES (?, ?, ?, ?)",
                    [(k, n, v, int(k in SCORED)) for n, (k, v) in enumerate(SHORT.items(), 1)])
    con.executemany("INSERT INTO decisions VALUES (?, ?, ?, ?)", decision_rows(S))
    con.commit()
    return con


def statements(text: str) -> list[str]:
    """The statements of one .sql file, in order (sqlite3 runs one statement per call)."""
    out, buf = [], ""
    for line in text.splitlines(True):
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    if buf.strip() and not all(ln.strip().startswith("--") or not ln.strip() for ln in buf.splitlines()):
        raise ValueError("a statement does not end with a semicolon")
    return out


def files() -> list[Path]:
    return sorted(SQL_DIR.glob("*.sql"))


def run_file(con: sqlite3.Connection, path: Path) -> list[tuple[list[str], list[tuple]]]:
    """Run every statement of one file; the result of each statement that returns rows, in order."""
    out = []
    for st in statements(path.read_text()):
        cur = con.execute(st)
        if cur.description is not None:
            out.append(([d[0] for d in cur.description], cur.fetchall()))
    return out


def run_all(con: sqlite3.Connection) -> dict[str, list]:
    return {p.name: run_file(con, p) for p in files()}


def _cell(x) -> str:
    if x is None:
        return "NULL"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def format_result(cols: list[str], rows: list[tuple], limit: int = ROWS) -> str:
    shown = rows if not limit or len(rows) <= limit else rows[:limit]
    cells = [[_cell(x) for x in r] for r in shown]
    width = [max([len(c)] + [len(r[k]) for r in cells]) for k, c in enumerate(cols)]
    line = lambda r: "  ".join(v.ljust(w) for v, w in zip(r, width)).rstrip()   # noqa: E731
    out = [line(cols), line(["-" * w for w in width])] + [line(r) for r in cells]
    if len(shown) < len(rows):
        k = len(rows) - len(shown)
        out.append(f"... {k} more row{'' if k == 1 else 's'}")
    return "\n".join(out)


def question(path: Path) -> str:
    """The opening comment of a file, up to its first blank comment line: the question it answers."""
    words = []
    for line in path.read_text().splitlines():
        text = line.removeprefix("--").strip()
        if not line.startswith("--") or not text:
            break
        words.append(text)
    return " ".join(words)


def file_output(results: list, limit: int = ROWS) -> str:
    return "\n\n".join(format_result(c, r, limit) for c, r in results)


def write_docs(results: dict) -> bool:
    """Rewrite the expected-output blocks of docs/sql.md; True if the file changed."""
    old = DOC.read_text()

    def sub(mt):
        body = "```text\n" + file_output(results[mt.group(2)]) + "\n```\n"
        return mt.group(1) + body + mt.group(4)

    new = DOC_BLOCK_RE.sub(sub, old)
    if new != old:
        DOC.write_text(new)
    return new != old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=ROWS, help="rows printed per result (0: all)")
    ap.add_argument("--db", default=":memory:", help="also keep the database in this new file (outside the repository)")
    ap.add_argument("--docs", action="store_true", help="rewrite the expected outputs in docs/sql.md")
    a = ap.parse_args()
    con = connect(a.db)
    results = run_all(con)
    for p in files():
        print(f"== sql/{p.name}: {question(p)}\n")
        print(file_output(results[p.name], a.rows) + "\n")
    if a.docs:
        print("docs/sql.md " + ("rewritten" if write_docs(results) else "unchanged"))
    if a.db != ":memory:":
        print(f"database kept in {a.db}", file=sys.stderr)


if __name__ == "__main__":
    main()
