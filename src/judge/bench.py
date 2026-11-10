"""Throughput and determinism check for concurrent requests to the one judge model.

Replays requests that are already in the response cache at several concurrency
levels. Nothing is written to the response cache: the replayed responses go to
results/cache/concurrency_replays.jsonl, and the report compares them with the
cached (sequential) responses: parsed decision, token probability, raw text.

python -m judge.bench --version v1 --n 12 --levels 1 2 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as C
from . import llm
from .data import load_records, load_splits
from .judgments import parse
from .prompts import build_messages, visible_narrative

REPLAY_FILE = C.RESULTS / "cache" / "concurrency_replays.jsonl"


def swap_used_mb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        return float(out.split("used = ")[1].split("M")[0])
    except Exception:
        return None


def _fetch(body: dict) -> dict:
    t0 = time.time()
    resp = llm._http("/api/chat", body, timeout=C.REQUEST_TIMEOUT_S)
    return {"content": resp["message"]["content"], "wall_s": round(time.time() - t0, 3),
            "logprobs": llm.compact_logprobs(resp.get("logprobs")),
            **{k: resp.get(k) for k in ("prompt_eval_count", "prompt_eval_cached_count", "eval_count",
                                        "prompt_eval_duration", "eval_duration")}}


def compare(replays: list[dict], cache: llm.Cache) -> list[dict]:
    """One row per (benchmark, concurrency level): throughput and differences
    from the cached sequential responses. Pure function of the two files."""
    groups: dict[tuple, list[dict]] = {}
    for r in replays:
        groups.setdefault((r["benchmark_at"], r["prompt_version"], r["concurrency"]), []).append(r)
    rows = []
    for (at, version, k), rs in sorted(groups.items()):
        ref = [parse(cache.records[llm.Cache.key(version, r["item_id"])]) for r in rs]
        raw = [cache.records[llm.Cache.key(version, r["item_id"])]["content"] for r in rs]
        new = [parse(r) for r in rs]
        dp = [abs(x["p_token"] - y["p_token"]) for x, y in zip(new, ref)
              if x["p_token"] == x["p_token"] and y["p_token"] == y["p_token"]]
        wall = rs[0]["level_wall_s"]
        rows.append({"benchmark_at": at, "version": version, "concurrency": k, "n_items": len(rs),
                     "level_wall_s": wall, "items_per_s": len(rs) / wall, "s_per_item": wall / len(rs),
                     "decisions_differ": sum(x["pred"] != y["pred"] or x["valid"] != y["valid"]
                                             for x, y in zip(new, ref)),
                     "fields_differ": sum(x["fields"] != y["fields"] for x, y in zip(new, ref)),
                     "raw_text_differs": sum(r["content"] != c for r, c in zip(rs, raw)),
                     "max_abs_delta_p_token": max(dp) if dp else None,
                     "swap_used_mb_before": rs[0]["swap_used_mb_before"],
                     "swap_used_mb_after": rs[0]["swap_used_mb_after"]})
    return rows


def load_replays() -> list[dict]:
    if not REPLAY_FILE.exists():
        return []
    return [json.loads(line) for line in REPLAY_FILE.read_text().splitlines() if line.strip()]


def main() -> None:
    from .run import log_event, now

    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()
    cache = llm.Cache()
    records, splits = load_records(), load_splits()
    ids = splits["dev"][: a.n]
    bodies = []
    for i in ids:
        messages, schema = build_messages(a.version, records[i], visible_narrative(records[i]["narrative"]))
        body = llm.request_body(messages, schema)
        if cache.get(llm.Cache.key(a.version, i), llm.request_hash(body)) is None:
            raise SystemExit(f"{a.version}|{i} is not cached")
        bodies.append(body)
    at = now()
    for k in a.levels:
        swap0 = swap_used_mb()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=k) as ex:
            got = list(ex.map(_fetch, bodies))
        wall = round(time.time() - t0, 2)
        swap1 = swap_used_mb()
        with open(REPLAY_FILE, "a", encoding="utf-8") as f:   # single writer, after the level
            for i, g in zip(ids, got):
                f.write(json.dumps({"benchmark_at": at, "prompt_version": a.version, "item_id": i,
                                    "concurrency": k, "level_wall_s": wall,
                                    "swap_used_mb_before": swap0, "swap_used_mb_after": swap1, **g},
                                   ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        row = [r for r in compare(load_replays(), cache) if r["benchmark_at"] == at and r["concurrency"] == k][0]
        print(json.dumps(row), flush=True)
    log_event({"event": "concurrency_benchmark", "benchmark_at": at, "version": a.version,
               "n_items": len(ids), "levels": a.levels, "replay_file": "results/cache/concurrency_replays.jsonl"})


if __name__ == "__main__":
    main()
