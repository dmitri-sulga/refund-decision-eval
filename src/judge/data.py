"""Build the frozen dataset.

Steps (all deterministic, recorded in data/provenance.json):
1. Stream the archived official bulk export (mirror, sha256-checked) and keep
   one decision type: product "Credit card", issue "Problem with a purchase
   shown on your statement", received 2023-01-01..2025-12-31, narrative present.
2. Download the same slice's metadata from the live official API (CSV export,
   no narratives) and keep only complaints whose product, sub-product, issue,
   sub-issue, company and company response are identical in both sources.
3. Map the company response to a binary label (config.LABEL_MAP); drop the rest.
4. Collapse near-duplicate narratives (cosine >= 0.5 between word-trigram TF-IDF
   vectors, connected components) to the earliest record of each group, so no
   re-filed or templated narrative appears in two splits. The similar pairs are
   kept for the label-consistency analysis.
5. Stratified (by label) seeded split into DEBUG / DEV / TEST / TRAIN.

Usage:  python -m judge.data --mirror /path/to/complaints.csv.zip
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import date, timedelta

import numpy as np

from . import config as C

csv.field_size_limit(sys.maxsize)

SUBSET_FILE = C.DATA / "mirror_subset.jsonl.gz"     # step 1 output
LIVE_FILE = C.DATA / "live_api_metadata.csv.gz"      # step 2 output
RECORDS_FILE = C.DATA / "records.jsonl.gz"           # frozen records used by every experiment
SPLITS_FILE = C.DATA / "splits.json"                 # frozen complaint IDs per split
PAIRS_FILE = C.DATA / "near_duplicate_pairs.json"    # for the label-consistency analysis
PROVENANCE_FILE = C.DATA / "provenance.json"


# ------------------------------------------------------------------ helpers
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_of(company_response: str):
    """Binary label from the recorded company response, or None (dropped)."""
    return C.LABEL_MAP.get(company_response)


def normalize_narrative(text: str) -> str:
    """Whitespace-only normalisation; the wording is never changed."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_jsonl_gz(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl_gz(path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ step 1
def extract_subset(zip_path) -> dict:
    digest = sha256_file(zip_path)
    if digest != C.MIRROR_SHA256:
        raise SystemExit(f"mirror sha256 mismatch: {digest}")
    n_total = n_slice = 0
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open("complaints.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for r in reader:
                n_total += 1
                if r["Product"] != C.PRODUCT or r["Issue"] != C.ISSUE:
                    continue
                if not (C.DATE_MIN <= r["Date received"] < C.DATE_MAX):
                    continue
                n_slice += 1
                if not r["Consumer complaint narrative"].strip():
                    continue
                rows.append({
                    "complaint_id": r["Complaint ID"],
                    "date_received": r["Date received"],
                    "product": r["Product"],
                    "sub_product": r["Sub-product"],
                    "issue": r["Issue"],
                    "sub_issue": r["Sub-issue"],
                    "company": r["Company"],
                    "company_response": r["Company response to consumer"],
                    "narrative": r["Consumer complaint narrative"],
                })
    rows.sort(key=lambda x: int(x["complaint_id"]))
    write_jsonl_gz(SUBSET_FILE, rows)
    return {"mirror_sha256": digest, "mirror_rows_total": n_total,
            "slice_rows": n_slice, "slice_rows_with_narrative": len(rows)}


# ------------------------------------------------------------------ step 2
def fetch_live_metadata() -> dict:
    # widen by two days: the live API stores UTC timestamps, the bulk file dates
    lo = (date.fromisoformat(C.DATE_MIN) - timedelta(days=2)).isoformat()
    hi = (date.fromisoformat(C.DATE_MAX) + timedelta(days=2)).isoformat()
    params = [("product", C.PRODUCT), ("issue", C.ISSUE), ("date_received_min", lo),
              ("date_received_max", hi), ("format", "csv"), ("no_aggs", "true")]
    url = C.LIVE_API_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=600) as resp:
        body = resp.read()
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(LIVE_FILE, "wb") as f:
        f.write(body)
    n = sum(1 for _ in csv.DictReader(io.StringIO(body.decode("utf-8"))))
    return {"live_api_url": url, "live_rows": n}


def load_live() -> dict:
    with gzip.open(LIVE_FILE, "rt", encoding="utf-8", newline="") as f:
        return {r["Complaint ID"]: r for r in csv.DictReader(f)}


# ------------------------------------------------------------------ step 4
def similar_pairs(texts: list[str], threshold: float) -> list[tuple[int, int, float]]:
    """Pairs (i < j) whose cosine similarity of word-trigram TF-IDF vectors is
    >= threshold. Trigrams shared by more than 1% of complaints are ignored, so
    the score measures shared distinctive wording (re-filed or templated text),
    not a shared topic."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(ngram_range=(3, 3), min_df=2, max_df=0.01, sublinear_tf=True,
                        dtype=np.float32).fit_transform(texts)
    out = []
    for start in range(0, X.shape[0], 4000):
        S = (X[start:start + 4000] @ X.T).tocoo()
        rows = S.row + start
        keep = (rows < S.col) & (S.data >= threshold)
        out += [(int(i), int(j), round(float(v), 4))
                for i, j, v in zip(rows[keep], S.col[keep], S.data[keep])]
    return sorted(out)


def components(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ------------------------------------------------------------------ step 5
def _allocate(n: int, class_sizes: dict) -> dict:
    """Largest-remainder allocation of n draws proportional to class sizes."""
    total = sum(class_sizes.values())
    raw = {k: n * v / total for k, v in class_sizes.items()}
    out = {k: int(np.floor(x)) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], -k), reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def stratified_split(ids: list[str], labels: list[int], sizes: dict, seed: int) -> dict:
    """sizes: ordered {split_name: n or None}; None takes the remainder."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for cid, y in sorted(zip(ids, labels), key=lambda t: int(t[0])):
        by_class.setdefault(y, []).append(cid)
    pools = {y: list(rng.permutation(v)) for y, v in sorted(by_class.items())}
    out = {}
    for name, n in sizes.items():
        remaining = {y: len(p) for y, p in pools.items()}
        take = remaining if n is None or n >= sum(remaining.values()) else _allocate(n, remaining)
        chosen = []
        for y in sorted(pools):
            chosen += pools[y][: take[y]]
            pools[y] = pools[y][take[y]:]
        out[name] = sorted(chosen, key=int)
    return out


# ------------------------------------------------------------------ main
def build(zip_path: str | None) -> None:
    prov = {"source": {
        "mirror_url": C.MIRROR_URL, "official_bulk_url": C.OFFICIAL_BULK_URL,
        "mirror_as_of": "2026-08-11", "live_api": C.LIVE_API_URL,
        "note": ("Consumer narratives were removed from the live CFPB database in "
                 "September 2026 (API release 24). Narratives come from an archived "
                 "copy of the official bulk export; labels and metadata are checked "
                 "against the live API."),
    }}
    if zip_path:
        prov["extract"] = extract_subset(zip_path)
    if not LIVE_FILE.exists():
        prov["live"] = fetch_live_metadata()

    subset = read_jsonl_gz(SUBSET_FILE)
    live = load_live()
    funnel = Counter()
    funnel["slice_with_narrative"] = len(subset)
    statuses = Counter(r["company_response"] for r in subset)
    kept = []
    fields = [("product", "Product"), ("sub_product", "Sub-product"), ("issue", "Issue"),
              ("sub_issue", "Sub-issue"), ("company", "Company"),
              ("company_response", "Company response to consumer")]
    for r in subset:
        lv = live.get(r["complaint_id"])
        if lv is None:
            funnel["dropped_not_in_live_api"] += 1
            continue
        if any(r[a] != lv[b] for a, b in fields):
            funnel["dropped_metadata_mismatch_vs_live_api"] += 1
            continue
        y = label_of(r["company_response"])
        if y is None:
            funnel["dropped_status_not_mapped"] += 1
            continue
        rec = dict(r)
        rec["label"] = y
        rec["narrative"] = normalize_narrative(r["narrative"])
        kept.append(rec)
    funnel["after_live_check_and_label_map"] = len(kept)

    pairs = similar_pairs([r["narrative"].lower() for r in kept], C.NEAR_DUP_COSINE)
    reps, n_multi = [], 0
    for g in components(len(kept), pairs):
        members = sorted(g, key=lambda i: (kept[i]["date_received"], int(kept[i]["complaint_id"])))
        reps.append(kept[members[0]])
        n_multi += len(members) > 1
    reps.sort(key=lambda r: int(r["complaint_id"]))
    funnel["near_duplicate_groups_size_gt1"] = n_multi
    funnel["records_removed_as_near_duplicates"] = len(kept) - len(reps)
    funnel["sampling_frame"] = len(reps)

    sizes = {"debug": C.N_DEBUG, "dev": C.N_DEV, "test": C.N_TEST, "train": C.N_TRAIN}
    splits = stratified_split([r["complaint_id"] for r in reps], [r["label"] for r in reps],
                              sizes, C.SPLIT_SEED)
    used = {cid for v in splits.values() for cid in v}
    records = [r for r in reps if r["complaint_id"] in used]
    write_jsonl_gz(RECORDS_FILE, records)

    # similar pairs with their labels, for the label-consistency (ceiling) analysis
    pair_rows = [{"a": kept[i]["complaint_id"], "b": kept[j]["complaint_id"], "cosine": v,
                  "same_company": kept[i]["company"] == kept[j]["company"],
                  "days_apart": abs((date.fromisoformat(kept[i]["date_received"])
                                     - date.fromisoformat(kept[j]["date_received"])).days),
                  "label_a": kept[i]["label"], "label_b": kept[j]["label"]}
                 for i, j, v in pairs]
    PAIRS_FILE.write_text(json.dumps(pair_rows, indent=0) + "\n")

    SPLITS_FILE.write_text(json.dumps({
        "seed": C.SPLIT_SEED, "stratified_by": "label",
        "sizes": {k: len(v) for k, v in splits.items()}, "ids": splits}, indent=1) + "\n")
    prov["funnel"] = dict(funnel)
    prov["status_counts_in_slice_with_narrative"] = dict(statuses)
    prov["label_map"] = C.LABEL_MAP
    prov["frame_base_rate"] = float(np.mean([r["label"] for r in reps]))
    old = json.loads(PROVENANCE_FILE.read_text()) if PROVENANCE_FILE.exists() else {}
    for k in ("extract", "live"):
        if k not in prov and k in old:
            prov[k] = old[k]
    PROVENANCE_FILE.write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps(prov, indent=1))


def load_records() -> dict:
    return {r["complaint_id"]: r for r in read_jsonl_gz(RECORDS_FILE)}


def load_splits() -> dict:
    return json.loads(SPLITS_FILE.read_text())["ids"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", help="path to the archived complaints.csv.zip")
    build(ap.parse_args().mirror)
