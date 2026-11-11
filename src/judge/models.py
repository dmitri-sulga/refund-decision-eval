"""Judge v1 and v2 on other local models: docs/analysis_plan_models.md and its Amendment 1.

Added after the held-out run and after the blind labels were locked. Sends only DEV and labelled-sample
complaints and refuses any held-out ID. Its cache, journal and run log live under results/models/; nothing is
written to results/cache/, results/run_log.jsonl or results/human_labels/. Every request is the frozen request of
judge v1 or v2 (prompts.py, llm.request_body) with the model replaced and three neutral penalties added.

python -m judge.models --verify                          # the checks `make check` runs; no contact with Ollama
python -m judge.models --run TAG                         # one model's four cells, in the registered order
python -m judge.models --run TAG --resend-unanswered     # after an interruption: send each unanswered request once
python -m judge.models --close-incomplete TAG            # after that one resend: the missing answers stay missing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
from datetime import datetime, timezone

from . import config as C
from . import freeze, llm
from .prompts import build_messages, visible_narrative

DIR = C.RESULTS / "models"
CACHE_FILE = DIR / "responses.jsonl"
JOURNAL = DIR / "sent.jsonl"
RUN_LOG = DIR / "run_log.jsonl"
ENV_FILE = DIR / "environment.json"
REGISTRATION = DIR / "registration.json"
LOG_EXTRACT = DIR / "server_log_before_first_request.txt"
OLLAMA_LOG = os.path.expanduser("~/.ollama/logs/server.log")
PLAN_FILE = C.ROOT / "docs" / "analysis_plan_models.md"

REGISTERED_COMMIT = "162909f4d589a3070910f3295a03ade2d2366714"    # the plan as registered (lines 1-149)
REGISTERED_LINES = 149
REGISTERED_SHA256 = "1232111927c56ae608712d58fcea942b34b4c5245028a9f7f41b93bc08021aba"
OLLAMA_VERSION = "0.40.15"
REFERENCE = ("qwen3.8:27b", "47951e4a666180bc2aedfc93c169472f2de783696fe196efbf063f5730ecf163")
# run order = the only order of every table (Amendment 1, section 9)
MODELS = (
    ("qwen3.5:9b", "d63f830f28d515fefea546f213fca7a75e9dd5176ef51a6deabc952d1c99a911"),
    ("gemma4:12b", "c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6"),
    ("phi4:14b", "ac896e5b8b34a1f4efa7b14d7520725140d5512484457fab45d2a4ea14c69dba"),
    ("mistral-small3.2:24b", "5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b"),
    ("granite4.1:30b", "3f3e5df8a021439fd6f867a0e526bdc303cac79c811201cb6bac193298cb9fcd"),          # Amendment 1
    ("glm-4.7-flash:latest", "4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6"),    # Amendment 1
    ("gemma4:31b", "92808281a855666acf35743d49004bb30c810152cd004baf78f31b4578920e49"),              # Amendment 1
)
DIGESTS = dict(MODELS)
VERSIONS = ("v1", "v2")
SPLITS = ("dev", "labelled")
CELLS = tuple((v, s) for v in VERSIONS for s in SPLITS)     # (v1, dev), (v1, labelled), (v2, dev), (v2, labelled)
NEUTRAL_PENALTIES = {"presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1}
OPTIONAL_FIELDS = {"think": ("think",), "logprobs": ("logprobs", "top_logprobs")}   # Amendment 1, section 5
REQUEST_FIELDS = ("model", "messages", "format", "think", "stream", "options", "logprobs", "top_logprobs",
                  "keep_alive", *C.OPTIONS, *NEUTRAL_PENALTIES)
HASHED = ("model", "messages", "format", "think", "options", "logprobs", "top_logprobs")
SHOW_FIELDS = ("parameters", "template", "system", "capabilities", "details", "modelfile")
MAX_PROMPT_TOKENS = C.OPTIONS["num_ctx"] - C.OPTIONS["num_predict"]     # 7,680
# files that must be byte-identical to the published code commit when the runner starts
PROTECTED = ("docs/analysis_plan_models.md", "src/judge/models.py", "src/judge/models_analysis.py",
             "src/judge/prompts.py", "src/judge/llm.py", "src/judge/judgments.py", "src/judge/config.py",
             "src/judge/metrics.py", "Makefile", "results/models/environment.json", "tests/test_models.py")


class Refusal(SystemExit):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def key(tag: str, version: str, item_id: str) -> str:
    return f"{tag}|{version}|{item_id}"


def split_key(k: str) -> tuple[str, str, str]:
    tag, version, item = k.rsplit("|", 2)
    return tag, version, item


def _jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log_event(event: dict) -> None:
    _append(RUN_LOG, {"at": now(), **event})


# ------------------------------------------------------------------ the registered text
def registered_part_sha256(path=None) -> str:
    lines = (path or PLAN_FILE).read_bytes().split(b"\n")
    return hashlib.sha256(b"\n".join(lines[:REGISTERED_LINES]) + b"\n").hexdigest()


# ------------------------------------------------------------------ which complaints may be sent
def item_sets() -> tuple[dict, set]:
    """{split: [ids]} for DEV and the labelled sample, and the held-out IDs. Refuses unless the held-out IDs are
    the frozen ones and neither allowed set touches them."""
    from .data import load_splits
    from .label import load_sample

    splits = load_splits()
    frozen = freeze.load()
    if list(freeze.heldout_ids_sha256()) != [frozen["heldout"]["n"], frozen["heldout"]["ids_sha256"]]:
        raise Refusal("the held-out IDs differ from the frozen ones; refusing.")
    sets = {"dev": list(splits["dev"]), "labelled": list(load_sample()["ids"])}
    test = set(splits["test"])
    if test & (set(sets["dev"]) | set(sets["labelled"])):
        raise Refusal("an allowed complaint is a held-out complaint; refusing.")
    return sets, test


def guard(item_id: str, allowed: set, test: set) -> None:
    """Called just before a request is journaled."""
    if item_id in test or item_id not in allowed:
        raise Refusal(f"complaint {item_id} is not a DEV or labelled-sample complaint; nothing was sent.")


# ------------------------------------------------------------------ requests
def build_body(tag: str, version: str, rec: dict, narrative: str, drop=()) -> dict:
    messages, schema = build_messages(version, rec, narrative)
    body = llm.request_body(messages, schema)
    body = {**body, "model": tag, "options": {**body["options"], **NEUTRAL_PENALTIES}}
    for name in drop:
        for f in OPTIONAL_FIELDS[name]:
            body.pop(f, None)
    return body


def request_hash(body: dict) -> str:
    """llm.request_hash, tolerating a field left out under Amendment 1, section 5."""
    canon = {k: body[k] for k in HASHED if k in body}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


class OllamaClient:
    """The four endpoints the runner uses; tests replace it with a fake."""

    def version(self) -> str:
        return llm._http("/api/version")["version"]

    def tags(self) -> dict:
        return {m["name"]: m["digest"] for m in llm._http("/api/tags")["models"]}

    def show(self, tag: str) -> dict:
        return llm._http("/api/show", {"model": tag})

    def chat(self, body: dict) -> dict:
        return llm._http("/api/chat", body, timeout=C.REQUEST_TIMEOUT_S)


def fetch(client, tag: str, version: str, item_id: str, body: dict, rhash: str, ollama_version: str) -> dict:
    requested_at = now()
    t0 = time.time()
    resp = client.chat(body)
    wall = time.time() - t0
    return {
        "key": key(tag, version, item_id), "model": tag, "prompt_version": version, "item_id": item_id,
        "request_sha256": rhash, "model_digest": DIGESTS[tag], "ollama_version": ollama_version,
        "response_model": resp.get("model"), "options": body["options"],
        "fields_sent": sorted(f for f in ("think", "logprobs", "top_logprobs") if f in body),
        "requested_at": requested_at, "wall_s": round(wall, 3), "content": resp["message"]["content"],
        "thinking": resp["message"].get("thinking"), "done_reason": resp.get("done_reason"),
        **{k: resp.get(k) for k in ("prompt_eval_count", "prompt_eval_cached_count", "eval_count",
                                    "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")},
        "logprobs": llm.compact_logprobs(resp.get("logprobs")),
    }


def classify(exc: BaseException, first_request: bool) -> tuple[str, str]:
    """Amendment 1, section 5. A refusal is an HTTP 4xx answer to a model's first request whose message names a
    request field: 'think' or 'logprobs' leave that field out for the model; any other named field drops the model.
    Everything else is an interruption."""
    if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and first_request:
        try:
            message = exc.read().decode("utf-8", "replace")
        except Exception:
            message = str(exc)
        named = [f for f in REQUEST_FIELDS if re.search(r"(?<![A-Za-z_])" + re.escape(f) + r"(?![A-Za-z_])", message)]
        for name, fields in OPTIONAL_FIELDS.items():
            if any(f in named for f in fields):
                return "drop_field:" + name, message
        if named:
            return "drop_model", message
        return "interruption", message
    return "interruption", f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------ state from the files
def state() -> dict:
    cache = {r["key"]: r for r in _jsonl(CACHE_FILE)}
    journal = _jsonl(JOURNAL)
    events = _jsonl(RUN_LOG)
    dropped_fields: dict[str, list] = {}
    status: dict[str, str] = {}
    for e in events:
        if e.get("event") == "field_dropped":
            dropped_fields.setdefault(e["model"], []).append(e["field"])
        elif e.get("event") in ("model_dropped", "closed_incomplete", "model_finished"):
            status[e["model"]] = e["event"]
    return {"cache": cache, "journal": journal, "dropped_fields": dropped_fields, "status": status}


def sends_by_key(journal: list[dict]) -> dict:
    out: dict[str, list[dict]] = {}
    for e in journal:
        out.setdefault(e["key"], []).append(e)
    return out


# ------------------------------------------------------------------ the public record and the tree
def git(*args) -> str:
    return subprocess.run(["git", "-C", str(C.ROOT), *args], capture_output=True, text=True, check=True).stdout


def github_main_head() -> str:
    """The commit GitHub shows as main (used only when a run starts)."""
    slug = load_registration()["repository"]
    out = subprocess.run(["gh", "api", f"repos/{slug}/git/refs/heads/main", "--jq", ".object.sha"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def load_registration() -> dict:
    if not REGISTRATION.exists():
        raise Refusal(f"results/models/{REGISTRATION.name} is missing: the code commit and the gist that record its "
                      "publication must be committed before the first request.")
    reg = json.loads(REGISTRATION.read_text())
    need = ("repository", "code_commit", "code_published_at", "gist_id", "gist_url", "gist_created_at")
    missing = [k for k in need if not reg.get(k)]
    if missing:
        raise Refusal(f"{REGISTRATION.name} lacks {', '.join(missing)}; refusing.")
    return reg


def preflight(tag: str, client, published_head=github_main_head) -> dict:
    """Everything that must hold before a model's first (or next) request. Returns what the run needs."""
    if tag not in DIGESTS:
        raise Refusal(f"{tag} is not a model of the plan; no model is ever added or replaced.")
    if registered_part_sha256() != REGISTERED_SHA256:
        raise Refusal("lines 1-149 of docs/analysis_plan_models.md differ from the registered text; refusing.")
    reg = load_registration()
    try:
        changed = git("diff", "--name-only", reg["code_commit"], "--", *PROTECTED).split()
    except subprocess.CalledProcessError:
        raise Refusal(f"the published code commit {reg['code_commit']} is not in this repository; refusing.")
    if changed:
        raise Refusal("these files differ from the published code commit: " + ", ".join(changed))
    if git("diff", "--name-only", "HEAD", "--", "docs/analysis_plan_models.md").strip():
        raise Refusal("docs/analysis_plan_models.md differs from the committed one; refusing.")
    dirty = [ln[3:] for ln in git("status", "--porcelain").splitlines()]
    outside = [p for p in dirty if not p.startswith("results/models/")]
    if outside:
        raise Refusal("uncommitted changes outside results/models/: " + ", ".join(outside))
    st = state()
    started = any(split_key(e["key"])[0] == tag for e in st["journal"])
    if dirty and not started:
        raise Refusal("results/models/ has uncommitted changes: commit and publish the previous model's results "
                      "before the next model starts.")
    head = git("rev-parse", "HEAD").strip()
    if published_head() != head:
        raise Refusal(f"HEAD {head[:12]} is not the commit GitHub shows as main: publish before sending.")
    if st["status"].get(tag):
        raise Refusal(f"{tag} is closed ({st['status'][tag]}); a finished model is never run again.")
    order = [t for t, _ in MODELS]
    for earlier in order[: order.index(tag)]:
        if not st["status"].get(earlier):
            raise Refusal(f"{earlier} comes before {tag} in the registered order and is not closed.")
    version = client.version()
    if version != OLLAMA_VERSION:
        raise Refusal(f"Ollama is {version}, registered {OLLAMA_VERSION}; nothing is sent under another version.")
    tags = client.tags()
    for t, d in ((tag, DIGESTS[tag]), REFERENCE):
        if tags.get(t) != d:
            raise Refusal(f"{t} has digest {tags.get(t)}, registered {d}; refusing.")
    env = json.loads(ENV_FILE.read_text())
    live = normalise_show(client.show(tag))
    for f in SHOW_FIELDS:
        if live.get(f) != env["models"][tag]["show"].get(f):
            raise Refusal(f"/api/show '{f}' of {tag} differs from results/models/environment.json; refusing.")
    from .label import is_locked
    if not is_locked("owner"):
        raise Refusal("the blind labels are not locked; refusing.")
    sets, test = item_sets()
    return {"sets": sets, "test": test, "ollama_version": version, "state": st, "registration": reg}


def normalise_show(show: dict) -> dict:
    """/api/show with the home directory written as '~', as in results/models/environment.json."""
    return json.loads(json.dumps(show).replace(json.dumps(os.path.expanduser("~"))[1:-1], "~"))


def write_log_extract() -> None:
    """Amendment 1, section 11: request paths and model loads from the Ollama server log, up to the first request."""
    if LOG_EXTRACT.exists() or not os.path.exists(OLLAMA_LOG):
        return
    keep = re.compile(r'"/api/|runner subprocess|msg="starting|loaded runners|Loaded tensors')
    lines = []
    with open(OLLAMA_LOG, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if keep.search(ln):
                lines.append(re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<ip>", ln.rstrip("\n")))
    LOG_EXTRACT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ the run
def work_list(tag: str, sets: dict, records: dict) -> list[tuple]:
    out = []
    for version, split in CELLS:
        for i in sets[split]:
            out.append((version, split, i, records[i], visible_narrative(records[i]["narrative"])))
    return out


def run_model(tag: str, client=None, resend_unanswered: bool = False, published_head=github_main_head,
              records=None) -> dict:
    from .data import load_records

    client = client or OllamaClient()
    pre = preflight(tag, client, published_head)
    st, sets, test = pre["state"], pre["sets"], pre["test"]
    allowed = set(sets["dev"]) | set(sets["labelled"])
    records = records if records is not None else load_records()
    sent = sends_by_key(st["journal"])
    unanswered = [k for k in sent if split_key(k)[0] == tag and k not in st["cache"]
                  and sent[k][-1].get("kind") != "refused"]
    if unanswered and not resend_unanswered:
        raise Refusal(f"{len(unanswered)} request(s) of {tag} were sent without an answer arriving: "
                      + ", ".join(sorted(unanswered)[:10]) + ". Resume with --resend-unanswered (each is sent once "
                      "more, and the report lists it).")
    if not st["journal"]:
        write_log_extract()
    drop = list(st["dropped_fields"].get(tag, []))
    first = not any(split_key(k)[0] == tag for k in st["cache"])
    log_event({"event": "run_started", "model": tag, "resend_unanswered": resend_unanswered})
    done = 0
    for version, split, item_id, rec, narrative in work_list(tag, sets, records):
        k = key(tag, version, item_id)
        if k in st["cache"]:
            continue
        previous = [e for e in sent.get(k, []) if e.get("kind") in ("send", "resend")]
        if previous and sent[k][-1].get("kind") != "refused":
            if any(e["kind"] == "resend" for e in previous):
                continue                      # sent twice already: stays unanswered (incomplete)
            kind = "resend"
        else:
            kind = "send"
        while True:
            body = build_body(tag, version, rec, narrative, drop)
            rhash = request_hash(body)
            guard(item_id, allowed, test)
            entry = {"at": now(), "key": k, "kind": kind, "request_sha256": rhash, "fields_dropped": sorted(drop)}
            _append(JOURNAL, entry)
            sent.setdefault(k, []).append(entry)
            try:
                r = fetch(client, tag, version, item_id, body, rhash, pre["ollama_version"])
            except BaseException as exc:          # noqa: BLE001 - every failure is classified and recorded
                verdict, message = classify(exc, first)
                if verdict.startswith("drop_field:"):
                    name = verdict.split(":", 1)[1]
                    _append(JOURNAL, {"at": now(), "key": k, "kind": "refused", "message": message[:500]})
                    log_event({"event": "field_dropped", "model": tag, "field": name, "message": message[:500]})
                    sent[k].append({"kind": "refused"})
                    drop.append(name)
                    kind = "send"
                    continue
                if verdict == "drop_model":
                    _append(JOURNAL, {"at": now(), "key": k, "kind": "refused", "message": message[:500]})
                    log_event({"event": "model_dropped", "model": tag, "message": message[:500]})
                    raise Refusal(f"{tag} refused its first request ({message[:200]}); it is dropped, not replaced.")
                log_event({"event": "run_stopped", "model": tag, "key": k, "reason": message[:500]})
                raise Refusal(f"interrupted at {k} ({message[:200]}). Nothing is dropped; resume with "
                              f"--run {tag} --resend-unanswered.") from exc
            if r["response_model"] not in (None, tag):
                raise Refusal(f"the server answered with model {r['response_model']} for {tag}; refusing.")
            _append(CACHE_FILE, r)
            st["cache"][k] = r
            first = False
            done += 1
            print(f"[{tag} {version} {split}] {item_id} {r['wall_s']:.1f}s in={r['prompt_eval_count']} "
                  f"out={r['eval_count']}", flush=True)
            break
    expected = {key(tag, v, i) for v, s in CELLS for i in sets[s]}
    missing = expected - set(st["cache"])
    if not missing:
        log_event({"event": "model_finished", "model": tag, "answers": len(expected)})
        print(f"{tag}: all {len(expected)} answers are in. Commit and publish results/models/ before the next model.")
    else:
        print(f"{tag}: {len(missing)} answer(s) missing after the one resend; close the model with "
              f"--close-incomplete {tag}.")
    return {"model": tag, "new_answers": done, "missing": len(missing)}


def close_incomplete(tag: str) -> None:
    st = state()
    sent = sends_by_key(st["journal"])
    mine = [k for k in sent if split_key(k)[0] == tag]
    if not mine:
        raise Refusal(f"{tag} has not started; there is nothing to close.")
    if st["status"].get(tag):
        raise Refusal(f"{tag} is already closed ({st['status'][tag]}).")
    not_resent = [k for k in mine if k not in st["cache"] and not any(e.get("kind") == "resend" for e in sent[k])]
    if not_resent:
        raise Refusal(f"{len(not_resent)} unanswered request(s) of {tag} have not had their one resend.")
    answered = sum(1 for k in st["cache"] if split_key(k)[0] == tag)
    log_event({"event": "closed_incomplete", "model": tag, "answers": answered})


# ------------------------------------------------------------------ the checks of `make check`
def verify() -> list[str]:
    """Amendment 1, section 11. Reads files only."""
    problems = []
    if registered_part_sha256() != REGISTERED_SHA256:
        problems.append("lines 1-149 of docs/analysis_plan_models.md differ from the registered text")
    if [t for t, _ in MODELS] != list(dict.fromkeys(t for t, _ in MODELS)) or REFERENCE[0] in DIGESTS:
        problems.append("the model table repeats a model or contains the reference model")
    if ENV_FILE.exists():
        env = json.loads(ENV_FILE.read_text())
        if env.get("ollama_version") != OLLAMA_VERSION:
            problems.append("environment.json records another Ollama version")
        for t, d in (*MODELS, REFERENCE):
            if env.get("models", {}).get(t, {}).get("digest") != d:
                problems.append(f"environment.json: digest of {t} differs from the table")
    else:
        problems.append("results/models/environment.json is missing")
    journal, cache = _jsonl(JOURNAL), _jsonl(CACHE_FILE)
    if not journal and not cache:
        return problems
    sets, test = item_sets()
    allowed = set(sets["dev"]) | set(sets["labelled"])
    reg = None
    try:
        reg = load_registration()
    except Refusal as e:
        problems.append(str(e))
    sent = sends_by_key(journal)
    for k, entries in sent.items():
        tag, version, item = split_key(k)
        if tag not in DIGESTS or version not in VERSIONS:
            problems.append(f"journal: {k} is not a registered model and prompt")
        if item in test or item not in allowed:
            problems.append(f"journal: {k} is not a DEV or labelled-sample complaint")
        kinds = [e.get("kind") for e in entries]
        sends = [x for x in kinds if x in ("send", "resend")]
        if kinds.count("resend") > 1 or len(sends) - kinds.count("resend") > 1 + kinds.count("refused"):
            problems.append(f"journal: {k} was sent more often than the rules allow ({kinds})")
    if reg and journal and min(e["at"] for e in journal) <= reg["code_published_at"]:
        problems.append("the first request was journaled before the code was published")
    seen = set()
    for r in cache:
        k = r["key"]
        tag, version, item = split_key(k)
        if k in seen:
            problems.append(f"cache: {k} appears twice")
        seen.add(k)
        if r.get("model") != tag or r.get("model_digest") != DIGESTS.get(tag):
            problems.append(f"cache: {k} records model {r.get('model')} / digest {r.get('model_digest')}")
        if r.get("response_model") not in (None, tag):
            problems.append(f"cache: {k} was answered by {r.get('response_model')}")
        if r.get("ollama_version") != OLLAMA_VERSION:
            problems.append(f"cache: {k} was answered under Ollama {r.get('ollama_version')}")
        if item in test:
            problems.append(f"cache: {k} concerns a held-out complaint")
        if any(r.get("options", {}).get(o) != v for o, v in NEUTRAL_PENALTIES.items()):
            problems.append(f"cache: {k} was sent without the neutral penalties")
        before = [e for e in sent.get(k, []) if e.get("kind") in ("send", "resend") and e["at"] <= r["requested_at"]
                  and e.get("request_sha256") == r["request_sha256"]]
        if not before:
            problems.append(f"cache: {k} has no journal entry before its request")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--run", metavar="TAG")
    g.add_argument("--close-incomplete", metavar="TAG")
    ap.add_argument("--resend-unanswered", action="store_true")
    a = ap.parse_args()
    if a.verify:
        problems = verify()
        if problems:
            sys.exit("models extension check failed:\n- " + "\n- ".join(problems))
        print("models extension check passed")
    elif a.run:
        print(json.dumps(run_model(a.run, resend_unanswered=a.resend_unanswered)))
    else:
        close_incomplete(a.close_incomplete)


if __name__ == "__main__":
    main()
