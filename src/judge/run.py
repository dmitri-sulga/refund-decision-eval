"""Run judge versions over a split, the gaming set, or the single held-out TEST run.

Resumable: every answer is cached the moment it arrives, so an interrupted job
restarts where it stopped and sends only the requests that have no cached answer.
Nothing is queued ahead of the model: at most `workers` requests are in flight
(one by default; see the concurrency check in the README). The first Ctrl-C or
SIGTERM stops sending, keeps the answer already on its way and then stops; a
second one abandons that answer too. An HTTP error or a timeout stops the run
at once. In the held-out run every request is written to a journal
(results/cache/sent.jsonl) before it is sent, so a resume can tell a request
that was sent without an answer from one never sent, and refuses to send it
again unless told to (--resend-unanswered), which the report then states.

python -m judge.run --version v1 --split debug          # smoke test / timing
python -m judge.run --version v1 --split dev
python -m judge.run --version v1 --gaming               # adversarial rewrites and controls of the gaming subset
python -m judge.run --final --dry-run                   # pre-flight checks of the held-out run, no model call
python -m judge.run --final                             # the one and only TEST run
python -m judge.run --final --resend-unanswered         # resume after a request was sent without an answer
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

from . import config as C
from . import freeze
from . import llm
from .data import load_records, load_splits
from .defence import filtered_item_id, sanitize
from .gaming import ATTACKS, CONTROLS, apply_attack, apply_control, attacked_item_id, gaming_ids
from .prompts import VERSIONS, build_messages, visible_narrative

TEST_MARKER = C.RESULTS / "test_run.json"
SENT_FILE = C.RESULTS / "cache" / "sent.jsonl"   # held-out run: every request, written before it is sent
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


# ------------------------------------------------------------------ sending
def journal_sent(path, key: str, pass_name: str) -> None:
    """Record a request before it is sent (append, flush, fsync)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"at": now(), "pass": pass_name, "key": key}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_journal(path=None) -> list[dict]:
    path = SENT_FILE if path is None else path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class StopRequest:
    """First Ctrl-C or SIGTERM: send nothing new, keep the answer already on its
    way, then stop. A second signal abandons that answer as well."""

    def __init__(self):
        self.requested, self._old = False, {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._old[sig] = signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\n{signal.Signals(signum).name}: no new request will be sent; waiting for the answer already on "
              "its way (send the signal again to abandon it).", flush=True)

    def __exit__(self, *exc):
        for sig, handler in self._old.items():
            signal.signal(sig, handler)
        return False


def send_each(todo: list, fetch_one, keep, workers: int = 1, before_send=None) -> None:
    """Send the requests in `todo` and hand every answer to `keep` as soon as it
    arrives. Nothing is queued: a request is submitted only when one of at most
    `workers` slots is free, so an error, a timeout or a stop request leaves no
    queued request behind to be sent, and every answer that still arrives is kept."""
    with StopRequest() as stop:
        if workers <= 1:
            for item in todo:
                if stop.requested:
                    break
                if before_send:
                    before_send(item)
                keep(fetch_one(item))
        else:
            pending, items = set(), iter(todo)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                try:
                    while True:
                        while len(pending) < workers and not stop.requested:
                            item = next(items, None)
                            if item is None:
                                break
                            if before_send:
                                before_send(item)
                            pending.add(ex.submit(fetch_one, item))
                        if not pending:
                            break
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        errors = [f.exception() for f in finished if f.exception() is not None]
                        for f in finished:
                            if f.exception() is None:
                                keep(f.result())
                        if errors:
                            raise errors[0]
                finally:                     # requests already sent: keep what still arrives
                    for f in wait(pending)[0]:
                        if f.exception() is None:
                            keep(f.result())
    if stop.requested:
        raise SystemExit("stopped on request: no further request was sent and every answer that arrived is "
                         "cached. The same command resumes with the requests that have no answer.")


def run(version: str, items, offline: bool = False, label: str = "", cache_tag: str = "",
        workers: int = 1, journal=None, pass_name: str = "scored") -> dict:
    """cache_tag stores the responses under '<version>@<tag>' (same prompt), which
    keeps a re-run apart from the scored responses. With workers > 1 the model
    calls run in threads; only this thread writes the cache (append, flush, fsync
    per record). With a journal every request is recorded before it is sent."""
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

    def keep(r):
        cache.put(r)
        done.append(r)
        eta = (len(todo) - len(done)) * (time.time() - t_start) / len(done) / 60
        print(f"[{label} {version} {len(done)}/{len(todo)}] {r['item_id']} {r['wall_s']:.1f}s "
              f"in={r['prompt_eval_count']} cached_in={r['prompt_eval_cached_count']} "
              f"out={r['eval_count']} eta<={eta:.1f}min", flush=True)
        if len(done) == min(SPEED_PROBE_CALLS, len(todo)):
            log_event({"event": "run_speed", "version": version, "label": label,
                       "workers": workers, **_speed(done)})

    def before_send(t):
        if journal is not None:
            journal_sent(journal, llm.Cache.key(key_version, t[0]), pass_name)

    try:
        send_each(todo, lambda t: llm.fetch(env, key_version, *t), keep, workers, before_send)
    except BaseException as e:
        log_event({"event": "run_stopped", "version": version, "label": label, "new_calls": len(done),
                   "remaining": len(todo) - len(done), "reason": type(e).__name__})
        raise
    elapsed = time.time() - t_start
    summary = {"version": version, "label": label, "n_items": len(items), "new_calls": len(done),
               "workers": workers, "new_calls_wall_s": round(sum(r["wall_s"] for r in done), 1),
               "elapsed_s": round(elapsed, 1)}
    if done:
        summary.update({k: v for k, v in _speed(done).items() if k != "calls"})
    print(json.dumps(summary), flush=True)
    return summary


ALLOWED_DIRTY = {"results/cache/responses.jsonl", "results/cache/replays.jsonl", "results/run_log.jsonl",
                 "results/test_run.json", "results/cache/sent.jsonl"}      # files the held-out run itself writes


def git_state() -> tuple[str, list[str]]:
    """HEAD commit and the paths that differ from it (the run must start from a commit)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(C.ROOT), *args], capture_output=True, text=True, check=True).stdout
    dirty = [line[3:] for line in git("status", "--porcelain").splitlines()]
    return git("rev-parse", "HEAD").strip(), dirty


def heldout_plan(frozen: dict, records: dict, splits: dict, cache: llm.Cache) -> list[dict]:
    """Every call of the scored held-out pass: each frozen judge version on every
    TEST complaint, plus v2 on the filtered text of complaints the input filter changes."""
    plan = []
    for v in frozen["final_versions"]:
        items = list(split_items("test", records, splits))
        if v == frozen["guarded_facts_version"]:
            items += list(filtered_items("test", records, splits))
        missing = sum(llm.Cache.key(v, i) not in cache.records for i, _, _ in items)
        plan.append({"version": v, "items": items, "n_items": len(items), "missing": missing})
    return plan


def unanswered_requests(path=None) -> list[str]:
    """Journaled requests with no cached answer: sent (or about to be sent) when
    the run stopped on an error, a timeout, a crash or a second stop signal."""
    from .replay import REPLAY_FILE

    answered = set(llm.Cache().records) | set(llm.Cache(REPLAY_FILE).records)
    return sorted({e["key"] for e in read_journal(path)} - answered)


def final_test_run(dry_run: bool = False, resend_unanswered: bool = False) -> None:
    """The single scored pass over TEST, followed by the registered noise pass.
    Refuses to start unless the method is frozen, unchanged and committed;
    refuses to start twice; an interrupted run resumes with the same frozen method,
    sending only requests without an answer, and refuses to send a journaled
    request a second time unless resend_unanswered is set."""
    from .replay import replay

    frozen = freeze.load()
    if frozen is None:
        sys.exit("results/frozen.json is missing: freeze the method first (python -m judge.freeze).")
    problems = freeze.verify_method(frozen)
    env = llm.environment()
    for k in ("ollama_version", "model_digest"):
        if env[k] != frozen["environment"][k]:
            problems.append(f"{k} is {env[k]}, frozen {frozen['environment'][k]}")
    commit, dirty = git_state()
    state = json.loads(TEST_MARKER.read_text()) if TEST_MARKER.exists() else None
    not_allowed = [p for p in dirty if p not in (ALLOWED_DIRTY if state else set())]
    if not_allowed:
        problems.append("uncommitted changes: " + ", ".join(not_allowed))
    if problems:
        sys.exit("refusing the held-out run:\n- " + "\n- ".join(problems))
    frozen_sha = freeze.sha256_path(freeze.FROZEN_FILE)
    if state is not None:
        if state.get("noise_finished_at"):
            sys.exit(f"TEST was already run at {state['started_at']}; refusing to run again.")
        if state["frozen_sha256"] != frozen_sha or state["commit"] != commit:
            sys.exit("an interrupted TEST run used a different frozen method or commit; refusing.")
    unanswered = unanswered_requests()
    if unanswered and not resend_unanswered:
        sys.exit(f"{len(unanswered)} held-out request(s) were sent without an answer arriving (an error, a timeout, "
                 "a crash or a second stop signal): " + ", ".join(unanswered) + ". Sending them again means the "
                 "model sees them twice; the report will say so. To do that, run: "
                 ".venv/bin/python -m judge.run --final --resend-unanswered && .venv/bin/python -m judge.report")
    n_test = len(load_splits()["test"])
    calls = len(frozen["final_versions"]) * n_test
    print(json.dumps({"held_out_complaints": n_test, "versions": frozen["final_versions"],
                      "workers": frozen["workers"], "commit": commit, "frozen_sha256": frozen_sha,
                      "scored_pass_calls": f"{calls} (plus one for every complaint the input filter changes)",
                      "noise_pass_calls": calls,
                      "expected_minutes": round(2 * calls * frozen["expected_s_per_call"] / 60),
                      "resuming": state is not None, "sent_without_answer": len(unanswered)}, indent=1),
          flush=True)
    if dry_run:
        print("dry run: checks passed, no model call was made and no held-out text was processed.")
        return
    records, splits = load_records(), load_splits()
    if state is None:
        state = {"started_at": now(), "versions": frozen["final_versions"], "workers": frozen["workers"],
                 "frozen_sha256": frozen_sha, "commit": commit}
        TEST_MARKER.write_text(json.dumps(state, indent=1) + "\n")
        log_event({"event": "test_run_started", "versions": frozen["final_versions"],
                   "workers": frozen["workers"], "frozen_sha256": frozen_sha, "commit": commit})
    if not state.get("finished_at"):
        for p in heldout_plan(frozen, records, splits, llm.Cache()):
            s = run(p["version"], p["items"], label="test", workers=frozen["workers"], journal=SENT_FILE,
                    pass_name="scored")
            log_event({"event": "run", "cache_tag": "", **s})
        state["finished_at"] = now()
        TEST_MARKER.write_text(json.dumps(state, indent=1) + "\n")
        log_event({"event": "test_run_finished", "versions": frozen["final_versions"]})
    for v in frozen["final_versions"]:          # registered noise pass: stored apart, never scored
        replay(v, splits["test"], records, label="test noise pass", journal=SENT_FILE)
    state["noise_finished_at"] = now()
    TEST_MARKER.write_text(json.dumps(state, indent=1) + "\n")
    log_event({"event": "test_noise_pass_finished", "versions": frozen["final_versions"]})


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
    ap.add_argument("--resend-unanswered", action="store_true",
                    help="with --final: send again the requests that were sent without an answer (reported)")
    ap.add_argument("--rerun-tag", default="", help="re-run into a separate cache namespace")
    ap.add_argument("--workers", type=int, default=1, help="concurrent model calls")
    a = ap.parse_args()
    if a.final:
        final_test_run(dry_run=a.dry_run, resend_unanswered=a.resend_unanswered)
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
