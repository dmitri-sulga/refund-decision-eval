"""Test-retest check: send cached requests again and keep the second answers apart.

Replays never go into the response cache (which stays the single source of every
scored number); they are appended to results/cache/replays.jsonl under the key
'<version>@replay|<item id>' and compared with the cached responses by the report.

python -m judge.replay --version v2            # every DEV complaint, sequential, resumable
python -m judge.replay --version v1 --limit 100   # the gaming subset first, then DEV order
"""
from __future__ import annotations

import argparse

from . import config as C
from . import llm
from .data import load_records, load_splits
from .prompts import build_messages, visible_narrative

REPLAY_FILE = C.RESULTS / "cache" / "replays.jsonl"
TAG = "replay"


def replay_key(version: str, item_id: str) -> str:
    return llm.Cache.key(f"{version}@{TAG}", item_id)


def replay(version: str, ids: list, records: dict, label: str = "replay") -> int:
    """Send the cached requests of `ids` again and store the answers in the replay file."""
    from .run import _speed, log_event

    cache, replays = llm.Cache(), llm.Cache(REPLAY_FILE)
    env = llm.environment()
    done = []
    for n, i in enumerate(ids, 1):
        messages, schema = build_messages(version, records[i], visible_narrative(records[i]["narrative"]))
        body = llm.request_body(messages, schema)
        rhash = llm.request_hash(body)
        if cache.get(llm.Cache.key(version, i), rhash) is None:
            raise SystemExit(f"{version}|{i} is not in the response cache")
        if replays.get(replay_key(version, i), rhash) is not None:
            continue
        r = llm.fetch(env, f"{version}@{TAG}", i, body, rhash)
        replays.put(r)
        done.append(r)
        print(f"[{label} {version} {n}/{len(ids)}] {i} {r['wall_s']:.1f}s", flush=True)
        if len(done) == 5:
            log_event({"event": "run_speed", "version": version, "label": label, "workers": 1, **_speed(done)})
    if done:
        log_event({"event": "replay", "version": version, "label": label, "n_items": len(ids),
                   "new_calls": len(done), "workers": 1,
                   **{k: v for k, v in _speed(done).items() if k != "calls"}})
    return len(done)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    records, splits = load_records(), load_splits()
    ids = splits["dev"]
    if a.limit:   # the gaming subset first (it needs a replay control), then DEV order
        from .gaming import gaming_ids
        first = gaming_ids(ids, {i: records[i]["label"] for i in ids})
        ids = (first + [i for i in ids if i not in set(first)])[: a.limit]
    replay(a.version, ids, records)


if __name__ == "__main__":
    main()
