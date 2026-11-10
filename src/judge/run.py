"""Run judge versions over a split, the gaming set, or the single held-out TEST run.

Resumable: every response is cached, so an interrupted job restarts where it
stopped. Calls are sequential by default (see the concurrency check in the README).

python -m judge.run --version v1 --split debug          # smoke test / timing
python -m judge.run --version v1 --split dev
python -m judge.run --version v1 --gaming               # adversarial rewrites and controls of the gaming subset
python -m judge.run --final --dry-run                   # pre-flight checks of the held-out run, no model call
python -m judge.run --final                             # the one and only TEST run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from . import config as C
from . import freeze
from . import llm
from .data import load_records, load_splits
from .defence import filtered_item_id, sanitize
from .gaming import ATTACKS, CONTROLS, apply_attack, apply_control, attacked_item_id, gaming_ids
from .prompts import VERSIONS, build_messages, visible_narrative

TEST_MARKER = C.RESULTS / "test_run.json"
SPEED_PROBE_CALLS = 5     # seconds per call are logged after this many new calls


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


def filtered_items(split: str, records: dict, splits: dict):
    """Unmodified complaints that the input filter flags and changes: the guarded
    pipeline reads the facts of the filtered text, so that text needs its own call."""
    for cid in splits[split]:
        rec = records[cid]
        text = visible_narrative(rec["narrative"])
        clean, flagged = sanitize(text)
        if flagged and clean != text:
            yield filtered_item_id(cid), rec, clean


def gaming_items(records: dict, splits: dict):
    labels = {i: records[i]["label"] for i in splits["dev"]}
    for cid in gaming_ids(splits["dev"], labels):
        rec = records[cid]
        vis = visible_narrative(rec["narrative"])
        for attack in ATTACKS:
            yield attacked_item_id(cid, attack), rec, apply_attack(vis, attack)
        for control in CONTROLS:
            yield attacked_item_id(cid, control), rec, apply_control(vis, control)


def _speed(recs: list[dict]) -> dict:
    gen_t = sum(r["eval_duration"] or 0 for r in recs) / 1e9
    prompt_t = sum(r["prompt_eval_duration"] or 0 for r in recs) / 1e9
    uncached = sum((r["prompt_eval_count"] or 0) - (r["prompt_eval_cached_count"] or 0) for r in recs)
    return {"calls": len(recs), "s_per_call": round(sum(r["wall_s"] for r in recs) / len(recs), 2),
            "generation_tokens_per_s": round(sum(r["eval_count"] or 0 for r in recs) / gen_t, 1) if gen_t else None,
            "prompt_tokens_per_s": round(uncached / prompt_t, 1) if prompt_t else None}


def run(version: str, items, offline: bool = False, label: str = "", cache_tag: str = "",
        workers: int = 1) -> dict:
    """cache_tag stores the responses under '<version>@<tag>' (same prompt), which
    keeps a re-run apart from the scored responses. With workers > 1 the model
    calls run in threads; only this thread writes the cache (append, flush, fsync
    per record)."""
    cache = llm.Cache()
    env = None if offline else llm.environment()
    items = list(items)
    key_version = f"{version}@{cache_tag}" if cache_tag else version
    todo = []
    for item_id, rec, narrative in items:
        messages, schema = build_messages(version, rec, narrative)
        body = llm.request_body(messages, schema)
        rhash = llm.request_hash(body)
        if cache.get(llm.Cache.key(key_version, item_id), rhash) is None:
            if env is None:
                raise RuntimeError(f"cache miss for {key_version}|{item_id} in offline mode")
            todo.append((item_id, body, rhash))
    t_start, done = time.time(), []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(llm.fetch, env, key_version, i, body, rhash) for i, body, rhash in todo]
        for fut in as_completed(futures):
            r = fut.result()
            cache.put(r)
            done.append(r)
            eta = (len(todo) - len(done)) * (time.time() - t_start) / len(done) / 60
            print(f"[{label} {version} {len(done)}/{len(todo)}] {r['item_id']} {r['wall_s']:.1f}s "
                  f"in={r['prompt_eval_count']} cached_in={r['prompt_eval_cached_count']} "
                  f"out={r['eval_count']} eta<={eta:.1f}min", flush=True)
            if len(done) == min(SPEED_PROBE_CALLS, len(todo)):
                log_event({"event": "run_speed", "version": version, "label": label,
                           "workers": workers, **_speed(done)})
    elapsed = time.time() - t_start
    summary = {"version": version, "label": label, "n_items": len(items), "new_calls": len(done),
               "workers": workers, "new_calls_wall_s": round(sum(r["wall_s"] for r in done), 1),
               "elapsed_s": round(elapsed, 1)}
    if done:
        summary.update({k: v for k, v in _speed(done).items() if k != "calls"})
    print(json.dumps(summary), flush=True)
    return summary


def heldout_plan(frozen: dict, records: dict, splits: dict, cache: llm.Cache) -> list[dict]:
    """Every call of the held-out run: each frozen judge version on every TEST
    complaint, plus v2 on the filtered text of complaints the input filter changes."""
    plan = []
    for v in frozen["final_versions"]:
        items = list(split_items("test", records, splits))
        if v == frozen["guarded_facts_version"]:
            items += list(filtered_items("test", records, splits))
        missing = sum(llm.Cache.key(v, i) not in cache.records for i, _, _ in items)
        plan.append({"version": v, "items": items, "n_items": len(items), "missing": missing})
    return plan


def final_test_run(dry_run: bool = False) -> None:
    """The single scored pass over TEST. Refuses to start unless the method is
    frozen and unchanged, refuses to start twice; an interrupted run resumes
    with the same frozen method."""
    frozen = freeze.load()
    if frozen is None:
        sys.exit("results/frozen.json is missing: freeze the method first (python -m judge.freeze).")
    problems = freeze.verify_method(frozen)
    env = llm.environment()
    for k in ("ollama_version", "model_digest"):
        if env[k] != frozen["environment"][k]:
            problems.append(f"{k} is {env[k]}, frozen {frozen['environment'][k]}")
    if problems:
        sys.exit("refusing the held-out run, the frozen method changed:\n- " + "\n- ".join(problems))
    frozen_sha = freeze.sha256_path(freeze.FROZEN_FILE)
    state = None
    if TEST_MARKER.exists():
        state = json.loads(TEST_MARKER.read_text())
        if state.get("finished_at"):
            sys.exit(f"TEST was already run at {state['started_at']}; refusing to run again.")
        if state["frozen_sha256"] != frozen_sha:
            sys.exit("an interrupted TEST run used a different frozen method; refusing.")
    records, splits = load_records(), load_splits()
    plan = heldout_plan(frozen, records, splits, llm.Cache())
    calls = sum(p["missing"] for p in plan)
    print(json.dumps({"held_out_complaints": len(splits["test"]),
                      "versions": frozen["final_versions"], "workers": frozen["workers"],
                      "calls_per_version": {p["version"]: p["n_items"] for p in plan},
                      "calls_to_make": calls, "already_cached": sum(p["n_items"] - p["missing"] for p in plan),
                      "expected_minutes": round(calls * frozen["expected_s_per_call"] / 60),
                      "resuming": state is not None}, indent=1), flush=True)
    if dry_run:
        print("dry run: checks passed, no model call was made.")
        return
    if state is None:
        state = {"started_at": now(), "versions": frozen["final_versions"],
                 "workers": frozen["workers"], "frozen_sha256": frozen_sha}
        TEST_MARKER.write_text(json.dumps(state, indent=1) + "\n")
        log_event({"event": "test_run_started", "versions": frozen["final_versions"],
                   "workers": frozen["workers"], "frozen_sha256": frozen_sha, "calls_to_make": calls})
    for p in plan:
        s = run(p["version"], p["items"], label="test", workers=frozen["workers"])
        log_event({"event": "run", "cache_tag": "", **s})
    state["finished_at"] = now()
    TEST_MARKER.write_text(json.dumps(state, indent=1) + "\n")
    log_event({"event": "test_run_finished", "versions": frozen["final_versions"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(VERSIONS))
    ap.add_argument("--split", choices=("debug", "dev"))
    ap.add_argument("--gaming", action="store_true")
    ap.add_argument("--labelled", action="store_true",
                    help="the human-labelled sample; allowed only after the blind labels are locked")
    ap.add_argument("--annotator", default="owner")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="with --final: checks only, no model call")
    ap.add_argument("--rerun-tag", default="", help="re-run into a separate cache namespace")
    ap.add_argument("--workers", type=int, default=1, help="concurrent model calls")
    a = ap.parse_args()
    if a.final:
        final_test_run(dry_run=a.dry_run)
        return
    records, splits = load_records(), load_splits()
    if a.labelled:
        from . import label as L
        if not L.is_locked(a.annotator):
            sys.exit("the blind labels are not locked: the judge may not see the labelled sample yet.")
        items = [(i, records[i], visible_narrative(records[i]["narrative"])) for i in L.load_sample()["ids"]]
        if a.version == freeze.GUARDED_FACTS_VERSION:
            for i, rec, text in list(items):
                clean, flagged = sanitize(text)
                if flagged and clean != text:
                    items.append((filtered_item_id(i), rec, clean))
    elif a.gaming:
        items = list(gaming_items(records, splits))
    else:
        items = list(split_items(a.split, records, splits))
        if a.split == "dev" and a.version == "v2":
            items += list(filtered_items("dev", records, splits))
    if a.limit:
        items = items[: a.limit]
    label = "labelled" if a.labelled else "gaming" if a.gaming else a.split
    s = run(a.version, items, label=label, cache_tag=a.rerun_tag, workers=a.workers)
    log_event({"event": "run", "cache_tag": a.rerun_tag, **s})


if __name__ == "__main__":
    main()
