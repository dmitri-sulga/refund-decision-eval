"""Run judge versions over a split, the gaming set, or the single held-out TEST run.

Resumable: every response is cached, so an interrupted job restarts where it
stopped. Calls are sequential on purpose (batching can change outputs).

python -m judge.run --version v1 --split debug          # smoke test / timing
python -m judge.run --version v1 --split dev
python -m judge.run --version v1 --gaming               # adversarial rewrites of the gaming subset
python -m judge.run --final                             # the one and only TEST run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from . import config as C
from . import llm
from .data import load_records, load_splits
from .gaming import ATTACKS, apply_attack, attacked_item_id, gaming_ids
from .prompts import VERSIONS, build_messages, visible_narrative

FINAL_VERSIONS = ("v1", "v2", "v2g")   # frozen before the TEST run
TEST_MARKER = C.RESULTS / "test_run.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(event: dict) -> None:
    C.RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(C.RUN_LOG, "a") as f:
        f.write(json.dumps({"at": now(), **event}) + "\n")


def split_items(split: str, records: dict, splits: dict):
    for cid in splits[split]:
        rec = records[cid]
        yield cid, rec, visible_narrative(rec["narrative"])


def gaming_items(records: dict, splits: dict):
    labels = {i: records[i]["label"] for i in splits["dev"]}
    for cid in gaming_ids(splits["dev"], labels):
        rec = records[cid]
        for attack in ATTACKS:
            yield attacked_item_id(cid, attack), rec, apply_attack(visible_narrative(rec["narrative"]), attack)


def run(version: str, items, offline: bool = False, label: str = "") -> dict:
    cache = llm.Cache()
    env = None if offline else llm.environment()
    items = list(items)
    t_start, n_new, new_wall = time.time(), 0, 0.0
    for k, (item_id, rec, narrative) in enumerate(items, 1):
        messages, schema = build_messages(version, rec, narrative)
        r, cached = llm.call(cache, env, version, item_id, messages, schema)
        if not cached:
            n_new += 1
            new_wall += r["wall_s"]
            left = len(items) - k
            eta = left * new_wall / n_new / 60
            print(f"[{label} {version} {k}/{len(items)}] {item_id} {r['wall_s']:.1f}s "
                  f"in={r['prompt_eval_count']} cached_in={r['prompt_eval_cached_count']} "
                  f"out={r['eval_count']} eta<={eta:.1f}min", flush=True)
    summary = {"version": version, "label": label, "n_items": len(items), "new_calls": n_new,
               "new_calls_wall_s": round(new_wall, 1), "elapsed_s": round(time.time() - t_start, 1)}
    print(json.dumps(summary), flush=True)
    return summary


def final_test_run() -> None:
    """The single scored pass over TEST. Refuses to start twice; an interrupted
    run resumes the same versions (no prompt may change in between)."""
    if TEST_MARKER.exists():
        state = json.loads(TEST_MARKER.read_text())
        if state.get("finished_at"):
            sys.exit(f"TEST was already run at {state['started_at']}; refusing to run again.")
        if tuple(state["versions"]) != FINAL_VERSIONS:
            sys.exit("an interrupted TEST run used different versions; refusing.")
    else:
        state = {"started_at": now(), "versions": list(FINAL_VERSIONS)}
        TEST_MARKER.write_text(json.dumps(state, indent=1) + "\n")
        log_event({"event": "test_run_started", "versions": list(FINAL_VERSIONS)})
    records, splits = load_records(), load_splits()
    for v in FINAL_VERSIONS:
        run(v, split_items("test", records, splits), label="test")
    state["finished_at"] = now()
    TEST_MARKER.write_text(json.dumps(state, indent=1) + "\n")
    log_event({"event": "test_run_finished", "versions": list(FINAL_VERSIONS)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(VERSIONS))
    ap.add_argument("--split", choices=("debug", "dev"))
    ap.add_argument("--gaming", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.final:
        final_test_run()
        return
    records, splits = load_records(), load_splits()
    items = list(gaming_items(records, splits) if a.gaming else split_items(a.split, records, splits))
    if a.limit:
        items = items[: a.limit]
    label = "gaming" if a.gaming else a.split
    s = run(a.version, items, label=label)
    log_event({"event": "run", **s})


if __name__ == "__main__":
    main()
