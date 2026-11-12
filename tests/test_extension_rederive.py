"""An independent re-derivation of the model extension's headline figures from the raw answers (eleventh post-run
change, after two reviews; written after the results were seen).

It shares no code with the registered analysis (models_analysis.py, hosted_analysis.py) or with extension_late.py:
it reads the answer files, the records and the person's labels directly, with its own JSON reading of each answer,
its own kappa and accuracy, and compares with results/models/analysis.json, results/models/hosted/analysis.json and
the late figures in results/metrics.json. The two comparators are taken from the frozen study code (baselines.py,
analysis.Systems with the frozen TF-IDF settings), as the plan fixes them: the issuer lookup's DEV decisions and the
TF-IDF model's DEV decisions, the latter checked against the confusion counts stored in results/metrics.json. Point estimates and counts only;
the intervals come from the registered bootstrap and are not recomputed here. Offline; no model is called."""
import gzip
import json

import numpy as np
import pytest

from judge import config as C

ROOT = C.ROOT
LOCAL = json.loads((C.RESULTS / "models" / "analysis.json").read_text())
HOSTED = json.loads((C.RESULTS / "models" / "hosted" / "analysis.json").read_text())
LATE = json.loads((C.RESULTS / "metrics.json").read_text())["extension"]["late"]


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _decision(content) -> tuple[int, bool]:
    """(money back, malformed): money back only for a JSON object whose monetary_relief is true."""
    try:
        d = json.loads(content)
    except (TypeError, ValueError):
        return 0, True
    v = d.get("monetary_relief") if isinstance(d, dict) else None
    return int(v is True), not isinstance(v, bool)


def _kappa(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    po = float((y == p).mean())
    pe = y.mean() * p.mean() + (1 - y.mean()) * (1 - p.mean())
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


@pytest.fixture(scope="module")
def world():
    with gzip.open(ROOT / "data" / "records.jsonl.gz", "rt", encoding="utf-8") as f:
        label = {r["complaint_id"]: int(r["label"]) for r in map(json.loads, f)}
    dev = json.loads((ROOT / "data" / "splits.json").read_text())["ids"]["dev"]
    lab = json.loads((ROOT / "data" / "label_sample.json").read_text())["ids"]
    person = {}
    for r in _rows(C.RESULTS / "human_labels" / "owner.jsonl"):        # the last record wins, an undo removes
        person.pop(r["complaint_id"], None)
        if not r.get("undo"):
            person[r["complaint_id"]] = r["label"]
    answers = {"local": {r["key"]: r for r in _rows(C.RESULTS / "models" / "responses.jsonl")},
               "hosted": {r["key"]: r for r in _rows(C.RESULTS / "models" / "hosted" / "responses.jsonl")},
               "study": {r["key"]: r for r in _rows(C.RESULTS / "cache" / "responses.jsonl")}}
    from judge import freeze                                 # the comparators, from the frozen study code
    from judge.analysis import Systems
    from judge.baselines import issuer_rate
    from judge.data import load_records, load_splits
    from judge.llm import Cache
    R, S = load_records(), load_splits()
    f, base, _ = issuer_rate(S["train"], R)
    assert list(S["dev"]) == dev
    tfidf = np.asarray(Systems(R, S, Cache(), lr_params=freeze.load()["tfidf_lr"]).predict("tfidf_lr", "dev")["pred"])
    y = np.array([label[i] for i in dev])
    stored = json.loads((C.RESULTS / "metrics.json").read_text())["dev"]["tfidf_lr"]["confusion"]
    mine = {f"{'relief' if a else 'none'}_{'relief' if b else 'none'}": int(((y == a) & (tfidf == b)).sum())
            for a in (0, 1) for b in (0, 1)}
    assert mine == stored, (mine, stored)
    return {"label": label, "dev": dev, "lab": lab, "person": person, "answers": answers,
            "lookup": np.array([int(f(R[i]) > base) for i in dev]), "tfidf": tfidf}


def _cell(w, arm, model, version, split, reference):
    ids = w["dev"] if split == "dev" else w["lab"]
    if model == reference:
        rows = [w["answers"]["study"].get(f"{version}|{i}") for i in ids]
    else:
        rows = [w["answers"][arm].get(f"{model}|{version}|{i}") for i in ids]
    assert all(r is not None for r in rows), (model, version, split)
    dec = [_decision(r["content"]) for r in rows]
    return ids, np.array([d for d, _ in dec]), sum(b for _, b in dec)


@pytest.mark.parametrize("arm, an", (("local", LOCAL), ("hosted", HOSTED)))
def test_counts_and_kappas_from_the_raw_answers(world, arm, an):
    reference = LOCAL["order"][0]
    total = 0
    for t in an["order"]:
        for k, c in an["models"][t]["cells"].items():
            version, split = k.split("|")
            ids, p, bad = _cell(world, arm, t, version, split, reference)
            y = np.array([world["label"][i] for i in ids])
            total += len(ids) if t != reference else 0
            assert (c["answered"], c["malformed"], c["pays"]) == (len(ids), bad, int(p.sum())), (t, k)
            assert np.isclose(c["kappa_vs_record"]["value"], _kappa(y, p)), (t, k)
            if split == "dev":
                d = _kappa(y, p) - _kappa(y, world["lookup"])
                reg = c.get("A", {}).get("readings", {}).get("as no money back") or c["A_descriptive"]
                assert np.isclose(reg["value"], d), (t, k)
    assert total == (4200 if arm == "local" else 1800)


@pytest.mark.parametrize("arm, an", (("local", LOCAL), ("hosted", HOSTED)))
def test_the_late_figures_from_the_raw_answers(world, arm, an):
    reference = LOCAL["order"][0]
    for t in an["order"]:
        for k, f in LATE["arms"][arm][t].items():
            version, split = k.split("|")
            ids, p, _ = _cell(world, arm, t, version, split, reference)
            y = np.array([world["label"][i] for i in ids])
            assert np.isclose(f["accuracy_vs_record"]["value"], (y == p).mean(), atol=1e-6), (t, k)
            assert np.isclose(f["payout_minus_record"]["value"], p.mean() - y.mean(), atol=1e-6), (t, k)
            if split == "dev":
                d = _kappa(y, p) - _kappa(y, world["tfidf"])
                assert np.isclose(f["minus_tfidf"]["kappa"]["value"], d, atol=1e-6), (t, k)
            else:
                per = world["person"]
                dec = [j for j, i in enumerate(ids) if per.get(i) in ("relief", "no_relief")]
                yd = np.array([int(per[ids[j]] == "relief") for j in dec])
                ya = np.array([int(per.get(i) == "relief") for i in ids])
                assert len(dec) == f["accuracy_vs_person_decided"]["n"]
                assert np.isclose(f["accuracy_vs_person_decided"]["value"], (yd == p[dec]).mean(), atol=1e-6)
                assert np.isclose(f["accuracy_vs_person_cannot_tell_as_no"]["value"], (ya == p).mean(), atol=1e-6)
